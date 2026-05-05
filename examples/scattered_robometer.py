"""Mock and Protocol for the scattered-frame robometer scorer.

The (future) real robometer endpoint will accept a full-length episode video and
return a small set of "important" frame indices with progress (and optionally
success) annotations only at those indices. We don't have that endpoint yet, so
this module ships:

    * ``ScatteredScoreResult``  — output dataclass.
    * ``ScatteredScorer``       — Protocol the real client will satisfy.
    * ``MockScatteredScorer``   — deterministic mock used by the training loop
      until the real endpoint exists, and by the pytest suite.

Frame 0 is ALWAYS pinned to progress 0 (per task spec), and the mock guarantees
``selected_indices`` is sorted, strictly increasing, and contained in [0, T).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class ScatteredScoreResult:
    """Per-video result from a scattered-frame robometer pass.

    ``selected_indices`` are env-step indices into the full-length video, sorted
    strictly increasing, all in [0, T). ``progress[k]`` and (optionally)
    ``success[k]`` annotate the state at ``selected_indices[k]``.
    """
    id: str
    selected_indices: np.ndarray
    progress: np.ndarray
    success: Optional[np.ndarray]


class ScatteredScorer(Protocol):
    """Duck-typed interface for any scattered-frame scorer."""

    def score(
        self,
        video: np.ndarray,
        task: str,
        sample_id: str,
        *,
        libero_rewards: Optional[np.ndarray] = None,
        libero_success_step: Optional[int] = None,
    ) -> ScatteredScoreResult:
        ...


_VALID_STRATEGIES = ("uniform_random", "uniform_grid", "libero_milestones")
_VALID_ORACLES = ("libero_reward_normalised", "monotone_random")


class MockScatteredScorer:
    """Deterministic stand-in for the scattered-frame robometer.

    Two orthogonal knobs:
      * ``strategy`` decides which env-step indices get annotated.
      * ``progress_oracle`` decides what progress value lands at those indices.

    All randomness is keyed by ``seed`` so two scorings of identical inputs
    produce identical outputs — important for the pytest suite.
    """

    def __init__(
        self,
        *,
        num_selected: int,
        strategy: str = "uniform_random",
        progress_oracle: str = "libero_reward_normalised",
        emit_success: bool = True,
        seed: int = 0,
    ) -> None:
        if num_selected < 2:
            raise ValueError(f"num_selected must be >= 2, got {num_selected}")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy={strategy!r} not in {_VALID_STRATEGIES}"
            )
        if progress_oracle not in _VALID_ORACLES:
            raise ValueError(
                f"progress_oracle={progress_oracle!r} not in {_VALID_ORACLES}"
            )
        self.num_selected = int(num_selected)
        self.strategy = strategy
        self.progress_oracle = progress_oracle
        self.emit_success = bool(emit_success)
        self.seed = int(seed)

    def _rng_for(self, sample_id: str) -> np.random.Generator:
        # Per-sample deterministic RNG so different episodes get different draws
        # but the same (seed, sample_id) always reproduces.
        h = abs(hash((self.seed, sample_id))) % (2**32)
        return np.random.default_rng(h)

    def _select_indices(
        self,
        T: int,
        rng: np.random.Generator,
        libero_rewards: Optional[np.ndarray],
    ) -> np.ndarray:
        K = min(self.num_selected, T)
        if self.strategy == "uniform_grid":
            # Equidistant indices including 0 and T-1 (reproduces the chunk
            # grid when K = T // query_freq + 1, useful for sanity checks).
            return np.linspace(0, T - 1, num=K, dtype=np.int64)
        if self.strategy == "uniform_random":
            # Always anchor 0; sample (K-1) more from [1, T) without
            # replacement, sort.
            if K == 1:
                return np.array([0], dtype=np.int64)
            tail = rng.choice(np.arange(1, T), size=K - 1, replace=False)
            return np.sort(np.concatenate([[0], tail])).astype(np.int64)
        if self.strategy == "libero_milestones":
            if libero_rewards is None:
                raise ValueError(
                    "strategy='libero_milestones' requires libero_rewards"
                )
            r = np.asarray(libero_rewards, dtype=np.float32)
            if r.shape != (T,):
                raise ValueError(
                    f"libero_rewards shape {r.shape} != ({T},)"
                )
            # "Milestones" = env-steps where libero reward jumps. Always
            # include frame 0; if we still don't have enough, top up with
            # uniform_random-style picks.
            jumps = np.where(np.diff(r, prepend=r[0]) > 0)[0]
            picks = sorted(set([0, *jumps.tolist()]))
            if len(picks) < K:
                pool = np.setdiff1d(np.arange(1, T), picks)
                if pool.size:
                    extra = rng.choice(
                        pool, size=min(K - len(picks), pool.size), replace=False
                    )
                    picks = sorted(set([*picks, *extra.tolist()]))
            return np.array(picks[:K], dtype=np.int64)
        raise AssertionError(f"unreachable strategy {self.strategy!r}")

    def _compute_progress(
        self,
        selected: np.ndarray,
        T: int,
        rng: np.random.Generator,
        libero_rewards: Optional[np.ndarray],
    ) -> np.ndarray:
        if self.progress_oracle == "libero_reward_normalised":
            if libero_rewards is None:
                raise ValueError(
                    "progress_oracle='libero_reward_normalised' requires libero_rewards"
                )
            r = np.asarray(libero_rewards, dtype=np.float32)
            if r.shape != (T,):
                raise ValueError(
                    f"libero_rewards shape {r.shape} != ({T},)"
                )
            cum = np.cumsum(np.clip(r, 0.0, None))
            denom = float(cum[-1]) if cum[-1] > 0.0 else 1.0
            normed = cum / denom
            out = normed[selected].astype(np.float32)
            out[0] = 0.0  # spec: frame 0 always 0
            return out
        if self.progress_oracle == "monotone_random":
            # Monotone non-decreasing in [0,1], anchored at 0.
            K = len(selected)
            raw = np.sort(rng.uniform(0.0, 1.0, size=K).astype(np.float32))
            raw[0] = 0.0
            return raw
        raise AssertionError(f"unreachable oracle {self.progress_oracle!r}")

    def _compute_success(
        self,
        selected: np.ndarray,
        libero_success_step: Optional[int],
    ) -> Optional[np.ndarray]:
        if not self.emit_success:
            return None
        out = np.zeros(len(selected), dtype=np.float32)
        if libero_success_step is None:
            return out
        out[selected >= int(libero_success_step)] = 1.0
        return out

    def score(
        self,
        video: np.ndarray,
        task: str,  # noqa: ARG002 — accepted to match the future real signature
        sample_id: str,
        *,
        libero_rewards: Optional[np.ndarray] = None,
        libero_success_step: Optional[int] = None,
    ) -> ScatteredScoreResult:
        if video.ndim != 4:
            raise ValueError(
                f"video must be 4D (T, H, W, C), got shape {video.shape}"
            )
        T = int(video.shape[0])
        if T < 2:
            raise ValueError(f"need at least 2 frames in the video, got T={T}")

        rng = self._rng_for(sample_id)
        selected = self._select_indices(T, rng, libero_rewards)
        progress = self._compute_progress(selected, T, rng, libero_rewards)
        success = self._compute_success(selected, libero_success_step)
        return ScatteredScoreResult(
            id=sample_id,
            selected_indices=selected,
            progress=progress,
            success=success,
        )
