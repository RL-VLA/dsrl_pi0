"""Scattered training-sample construction.

Pipeline (see ``docs/scattered_training_samples_plan.md``):

    ScatteredScoreResult     ─►   interpolate_progress / interpolate_success
                                            │
                                            ▼
                              ScatteredSampleConstructor (mode-dependent)
                                            │
                                            ▼
              List[ScatteredSample]  →  _apply_reward_fn_scattered
                                            │
                                            ▼
                            (rewards, masks) per sample, ready for the buffer

The reward functions in ``examples/train_utils_sim_robometer.py`` are reused
unchanged: we package the per-sample (progress_pre, progress_post, success_pre,
success_post) into a length-(K+1) synthetic ``ScoreResult`` so
``_post_chunk_shift`` produces ``progress_post`` exactly. That keeps a single
source of reward semantics — scattered mode just changes WHICH frames produce
samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def interpolate_progress(
    selected_indices: np.ndarray,
    progress_at_selected: np.ndarray,
    T: int,
) -> np.ndarray:
    """Dense per-env-step progress curve from sparse annotated points.

    Pins frame 0 to 0.0 (per task spec). Linear interpolation between adjacent
    anchors. Right-edge held constant past the last annotation (no
    extrapolation).
    """
    sel = np.asarray(selected_indices, dtype=np.int64)
    val = np.asarray(progress_at_selected, dtype=np.float32)
    if sel.shape != val.shape:
        raise ValueError(
            f"selected_indices {sel.shape} != progress_at_selected {val.shape}"
        )
    if sel.ndim != 1:
        raise ValueError(f"selected_indices must be 1D, got shape {sel.shape}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if sel.size == 0:
        # No annotations — flat zero curve.
        return np.zeros((T,), dtype=np.float32)
    if np.any(np.diff(sel) <= 0):
        raise ValueError("selected_indices must be strictly increasing")
    if sel[0] < 0 or sel[-1] >= T:
        raise ValueError(
            f"selected_indices out of bounds for T={T}: min={sel[0]}, max={sel[-1]}"
        )

    # Anchor frame 0 to 0.0 if not already present.
    if sel[0] != 0:
        sel = np.concatenate([[0], sel])
        val = np.concatenate([[np.float32(0.0)], val])
    else:
        # Force progress 0 at frame 0 even if the scorer disagreed — task
        # spec is explicit.
        val = val.copy()
        val[0] = 0.0

    out = np.empty((T,), dtype=np.float32)
    # Linear interpolation between anchors.
    out[: sel[-1] + 1] = np.interp(
        np.arange(sel[-1] + 1), sel, val
    ).astype(np.float32)
    if sel[-1] + 1 < T:
        # Right-edge: hold constant.
        out[sel[-1] + 1 :] = val[-1]
    return out


def interpolate_success(
    selected_indices: np.ndarray,
    success_at_selected: np.ndarray,
    T: int,
) -> np.ndarray:
    """Dense per-env-step success curve.

    Step semantics: ``success_dense[t] = max(success_at_selected[k] for k where
    selected_indices[k] <= t)``. Never decreases — success is a one-way event.
    Frame 0 takes the value of the first annotation if it sits at index 0,
    otherwise 0.
    """
    sel = np.asarray(selected_indices, dtype=np.int64)
    val = np.asarray(success_at_selected, dtype=np.float32)
    if sel.shape != val.shape:
        raise ValueError(
            f"selected_indices {sel.shape} != success_at_selected {val.shape}"
        )
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    if sel.size == 0:
        return np.zeros((T,), dtype=np.float32)
    if np.any(np.diff(sel) <= 0):
        raise ValueError("selected_indices must be strictly increasing")

    out = np.zeros((T,), dtype=np.float32)
    running = 0.0
    j = 0
    for t in range(T):
        while j < sel.size and sel[j] <= t:
            running = max(running, float(val[j]))
            j += 1
        out[t] = running
    return out


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScatteredSample:
    """One reconstructed (s, a, s', r) tuple at an off-grid position.

    ``noise`` is shape ``(query_freq, noise_dim)`` — sliced (or padded) from
    the per-env-step lifted-noise stream. ``executed`` is the matching env-action
    chunk. ``progress_pre/post`` and ``success_pre/post`` are the values
    eventually fed to the reward fn.
    """
    start: int                     # env-step at the start of this sample's chunk
    end: int                       # exclusive end (start + query_freq); may exceed T (right-padded)
    obs: dict                      # SAC obs at ``start``
    next_obs: dict                 # SAC obs at min(end, T-1) — used for next_observations
    noise: np.ndarray              # (H, noise_dim)
    executed: np.ndarray           # (H, env_action_dim)
    progress_pre: float
    progress_post: float
    success_pre: float
    success_post: float
    padded: bool                   # True if any chunk slice required edge padding


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

# env -> (noise_pad_mode, action_pad_mode)
#   "zero"   pad with zeros (relative/delta-action envs)
#   "repeat" repeat the last/first valid value (absolute-action envs)
_ENV_PAD_MODES: Dict[str, Tuple[str, str]] = {
    "libero": ("repeat", "zero"),       # noise repeat-last; libero env-actions are deltas
    "aloha_cube": ("repeat", "zero"),   # same convention as libero in this repo
}


def _pad_modes_for(env: str) -> Tuple[str, str]:
    for prefix, modes in _ENV_PAD_MODES.items():
        if env.startswith(prefix):
            return modes
    raise ValueError(
        f"unknown env={env!r}; add a pad-mode entry to scattered_samples._ENV_PAD_MODES"
    )


def _pad_chunk(
    stream: np.ndarray,
    start: int,
    end: int,
    *,
    pad_mode: str,
) -> np.ndarray:
    """Slice ``stream[start:end]``, padding off-edge entries.

    ``pad_mode='zero'``: out-of-range positions get a zeros-row.
    ``pad_mode='repeat'``: out-of-range positions copy the nearest in-range row.
    The result always has length ``end - start``.
    """
    T = stream.shape[0]
    H = end - start
    if H <= 0:
        raise ValueError(f"empty chunk: start={start}, end={end}")
    out_shape = (H,) + stream.shape[1:]
    out = np.zeros(out_shape, dtype=stream.dtype)
    for i, t in enumerate(range(start, end)):
        if 0 <= t < T:
            out[i] = stream[t]
        else:
            if pad_mode == "zero":
                pass  # already zeros
            elif pad_mode == "repeat":
                clamped = max(0, min(T - 1, t))
                out[i] = stream[clamped]
            else:
                raise ValueError(f"unknown pad_mode={pad_mode!r}")
    return out


# ---------------------------------------------------------------------------
# Sample constructors
# ---------------------------------------------------------------------------

def _obs_at(obs_stream: Sequence[dict], idx: int, T: int) -> dict:
    clamped = max(0, min(T - 1, idx))
    return obs_stream[clamped]


def _build_sample(
    *,
    start: int,
    H: int,
    obs_stream: Sequence[dict],
    noise_stream: np.ndarray,
    executed_stream: np.ndarray,
    progress_dense: np.ndarray,
    success_dense: np.ndarray,
    env: str,
) -> ScatteredSample:
    T = noise_stream.shape[0]
    end = start + H
    noise_pad, action_pad = _pad_modes_for(env)
    noise_chunk = _pad_chunk(noise_stream, start, end, pad_mode=noise_pad)
    exec_chunk = _pad_chunk(executed_stream, start, end, pad_mode=action_pad)

    pre_idx = max(0, min(T - 1, start))
    post_idx = max(0, min(T - 1, end))

    return ScatteredSample(
        start=int(start),
        end=int(end),
        obs=_obs_at(obs_stream, start, T),
        next_obs=_obs_at(obs_stream, end, T),
        noise=noise_chunk,
        executed=exec_chunk,
        progress_pre=float(progress_dense[pre_idx]),
        progress_post=float(progress_dense[post_idx]),
        success_pre=float(success_dense[pre_idx]),
        success_post=float(success_dense[post_idx]),
        padded=bool(start < 0 or end > T),
    )


def construct_look_future(
    *,
    selected_indices: np.ndarray,
    obs_stream: Sequence[dict],
    noise_stream: np.ndarray,
    executed_stream: np.ndarray,
    progress_dense: np.ndarray,
    success_dense: np.ndarray,
    H: int,
    env: str,
) -> List[ScatteredSample]:
    """For each selected frame f, build (s[f], a[f:f+H]).

    Discards candidates with ``f >= T - 1`` (no future to look at).
    """
    T = noise_stream.shape[0]
    out: List[ScatteredSample] = []
    for f in selected_indices:
        f = int(f)
        if f >= T - 1:
            continue
        out.append(
            _build_sample(
                start=f,
                H=H,
                obs_stream=obs_stream,
                noise_stream=noise_stream,
                executed_stream=executed_stream,
                progress_dense=progress_dense,
                success_dense=success_dense,
                env=env,
            )
        )
    return out


def construct_look_history(
    *,
    selected_indices: np.ndarray,
    obs_stream: Sequence[dict],
    noise_stream: np.ndarray,
    executed_stream: np.ndarray,
    progress_dense: np.ndarray,
    success_dense: np.ndarray,
    H: int,
    env: str,
) -> List[ScatteredSample]:
    """For each selected frame f, build (s[f-H], a[f-H:f]).

    When f - H < 0 we left-pad. Discards f == 0 (no history).
    """
    out: List[ScatteredSample] = []
    for f in selected_indices:
        f = int(f)
        if f == 0:
            continue
        start = f - H
        # Note: _build_sample handles negative start via _pad_chunk.
        out.append(
            _build_sample(
                start=start,
                H=H,
                obs_stream=obs_stream,
                noise_stream=noise_stream,
                executed_stream=executed_stream,
                progress_dense=progress_dense,
                success_dense=success_dense,
                env=env,
            )
        )
    return out


def construct_random_subsample(
    *,
    obs_stream: Sequence[dict],
    noise_stream: np.ndarray,
    executed_stream: np.ndarray,
    progress_dense: np.ndarray,
    success_dense: np.ndarray,
    H: int,
    env: str,
    n: int,
    rng: np.random.Generator,
) -> List[ScatteredSample]:
    """Pick ``n`` random in-range starts (no padding required) and build samples."""
    T = noise_stream.shape[0]
    if T <= H:
        raise ValueError(f"trajectory length {T} must be > query_freq {H}")
    pool = np.arange(0, T - H + 1)
    if n > pool.size:
        n = int(pool.size)
    starts = np.sort(rng.choice(pool, size=n, replace=False))
    return [
        _build_sample(
            start=int(s),
            H=H,
            obs_stream=obs_stream,
            noise_stream=noise_stream,
            executed_stream=executed_stream,
            progress_dense=progress_dense,
            success_dense=success_dense,
            env=env,
        )
        for s in starts
    ]


# ---------------------------------------------------------------------------
# Reward-fn reuse
# ---------------------------------------------------------------------------
#
# Scattered samples do NOT lie on a contiguous chunk grid, so we can't pack
# them into a single length-K ScoreResult and let the reward fn diff
# adjacent entries — that would compute ``progress_pre[k+1] - progress_pre[k]``
# instead of the per-sample ``progress_post[k] - progress_pre[k]`` we need.
#
# Instead we build one length-2 ``[pre, post]`` synthetic ScoreResult **per
# sample** and call the reward fn once with B=K, T=2. The reward at position
# ``[b, 0]`` is then exactly the per-sample reward for the kinds that diff
# (``progress_delta``) or post-shift (``progress``, ``success``).
#
# For "sparse base + progress" reward kinds (``libero_*``, ``robo_success_*``),
# the per-chunk-grid version replaces ``-1`` with ``0`` at the terminal index
# and zeros the terminal mask. With T=2 that modification lands at index 1
# (which we discard). So we post-process: locate the sample whose chunk
# straddles ``libero_success_step``, set its mask to 0, and — for kinds in
# ``_TERMINAL_CREDIT_KINDS`` — bump its reward by +1 (replacing the sparse
# ``-1`` base with ``0``).

@dataclass
class _PerSampleScore:
    """A length-2 [pre, post] synthetic ScoreResult for a single scattered sample."""
    progress: np.ndarray             # (2,) float32 — [progress_pre, progress_post]
    success: Optional[np.ndarray]    # (2,) float32 — [success_pre, success_post] or None


# Reward kinds whose chunk-grid implementation replaces the sparse base
# ``-1`` with ``0`` at the terminal step. For scattered samples we re-apply
# that bump on the straddling sample.
_TERMINAL_CREDIT_KINDS = frozenset({
    "libero_sparse",
    "libero_success_plus_robo_progress",
    "robo_success_sparse",
    "robo_success_plus_robo_progress",
})


def _per_sample_score(
    sample: ScatteredSample, *, emit_success: bool
) -> _PerSampleScore:
    progress = np.array(
        [sample.progress_pre, sample.progress_post], dtype=np.float32
    )
    if emit_success:
        success = np.array(
            [sample.success_pre, sample.success_post], dtype=np.float32
        )
    else:
        success = None
    return _PerSampleScore(progress=progress, success=success)


def _find_straddling_sample(
    samples: Sequence[ScatteredSample], success_step: int
) -> Optional[int]:
    for k, s in enumerate(samples):
        if s.start <= success_step < s.end:
            return k
    return None


def apply_reward_fn_scattered(
    reward_fn: Callable,
    samples: Sequence[ScatteredSample],
    *,
    libero_is_success: bool,
    libero_success_step: Optional[int],
    H: int,                          # noqa: ARG001 — kept for symmetry / future use
    robometer_success_threshold: float,
    emit_success: bool,
    reward_kind: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a per-chunk-grid reward fn against scattered samples.

    Returns ``(rewards, masks)`` of length ``K = len(samples)``.

    ``libero_is_success`` is the EPISODE-level flag. The terminal mask is
    re-targeted to the sample whose chunk STRADDLES ``libero_success_step``
    (fallback: the last sample). ``reward_kind`` is needed to know whether the
    reward fn applies a sparse-base terminal credit that we must re-apply on
    the straddling sample (since we always read position 0 of the per-sample
    length-2 reward, not position -1).
    """
    K = len(samples)
    if K == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    scores = [_per_sample_score(s, emit_success=emit_success) for s in samples]
    metas = [
        {
            "query_steps": 2,
            # Always False here — terminal handling is done downstream.
            "libero_is_success": False,
            "robometer_success_threshold": float(robometer_success_threshold),
        }
        for _ in samples
    ]
    rewards_BT, masks_BT = reward_fn(scores, metas)
    rewards_BT = np.asarray(rewards_BT, dtype=np.float32)
    masks_BT = np.asarray(masks_BT, dtype=np.float32)
    if rewards_BT.shape != (K, 2) or masks_BT.shape != (K, 2):
        raise ValueError(
            f"reward_fn returned shapes rewards={rewards_BT.shape}, masks={masks_BT.shape}; "
            f"expected ({K}, 2) for both"
        )
    rewards = rewards_BT[:, 0].copy()
    masks = masks_BT[:, 0].copy()

    if libero_is_success:
        if libero_success_step is not None:
            straddling = _find_straddling_sample(samples, int(libero_success_step))
        else:
            straddling = None
        if straddling is None:
            straddling = K - 1
        masks[straddling] = 0.0
        if reward_kind is not None and reward_kind in _TERMINAL_CREDIT_KINDS:
            # Replace the sparse base -1 with 0 at the straddling sample
            # (matches what the chunk-grid version does at the terminal step).
            rewards[straddling] = rewards[straddling] + 1.0
    return rewards, masks
