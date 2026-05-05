from tqdm import tqdm
import numpy as np
import time
import wandb
import jax
from openpi_client import image_tools
import math
import PIL
import warnings
from concurrent.futures import Future
from typing import Callable, Dict, List, Optional, Tuple

from robometer_client import RobometerClient, VideoSample, ScoreResult


# ============================================================================
# Robometer reward plumbing
# ============================================================================
#
# Reward functions are BATCHED and SELF-CONTAINED by design:
#
#   - Each fn accepts a list of ``ScoreResult`` objects and a list of meta dicts,
#     both of length B (one entry per episode in the batch).
#   - Returns ``(rewards, masks)`` as two (B, query_steps) float32 arrays.
#   - No fn ever calls another reward fn as a fallback — if a prerequisite is
#     missing (e.g. the server gave us no success head but the reward requires
#     one) the fn raises. The training loop's fail_behavior decides whether to
#     propagate or drop the offending episode BEFORE the batch is assembled.
#   - Every fn validates its own inputs and prints a clear error on mismatch.
#     This makes "silently falling back" impossible.
#
# The callsite in ``trajwise_alternating_training_loop`` is responsible for:
#   1. draining futures into a batch of (traj, ScoreResult) pairs,
#   2. dropping/raising on individual future failures per fail_behavior,
#   3. building the per-episode meta dicts,
#   4. calling the reward fn ONCE on the full batch.
#
# ----------------------------------------------------------------------------
# Off-by-one indexing convention (IMPORTANT):
#
# ``chunk_frames[k]`` is captured BEFORE chunk k executes (env-step k*query_freq).
# So the server's ``progress[k]`` / ``success[k]`` describe the state at the
# START of chunk k, i.e. the OUTCOME of chunk k-1.
#
# In the SAC transition (s_k, a_k, s_{k+1}) the reward attached to action a_k
# should reflect "what happened during chunk k" — that's the change between
# state-before-chunk-k (frame k) and state-after-chunk-k (frame k+1).
#
# Therefore reward fns that index into progress/success per chunk MUST use
# ``progress[k+1]`` (or ``progress_post[k]`` from ``_post_chunk_shift`` below)
# as the score attributed to chunk k. The last chunk (k = T-1) has no
# post-chunk frame in our buffer, so we reuse the last available value.
# ----------------------------------------------------------------------------

# (scores: List[ScoreResult], metas: List[Dict]) -> (rewards: (B, T), masks: (B, T))
RewardFn = Callable[[List[ScoreResult], List[Dict]], Tuple[np.ndarray, np.ndarray]]


def _validate_batch(scores: List[ScoreResult], metas: List[Dict]) -> Tuple[int, int]:
    """Shared pre-flight check. Returns (B, T)."""
    B = len(scores)
    if B == 0:
        raise ValueError("reward fn received an empty batch")
    if len(metas) != B:
        raise ValueError(f"len(scores)={B} != len(metas)={len(metas)}")
    T = int(metas[0]["query_steps"])
    if T <= 0:
        raise ValueError(f"meta[0].query_steps must be positive, got {T}")
    for b, m in enumerate(metas):
        if int(m["query_steps"]) != T:
            raise ValueError(
                f"inconsistent query_steps in batch: meta[0]={T}, meta[{b}]={m['query_steps']}"
            )
    return B, T


def _require_progress_len(scores: List[ScoreResult], T: int) -> np.ndarray:
    """Stack scores[b].progress into a (B, T) array; raise if any episode is off."""
    out = np.empty((len(scores), T), dtype=np.float32)
    for b, s in enumerate(scores):
        if s is None:
            raise ValueError(f"expected ScoreResult at batch index {b}, got None")
        if s.progress is None:
            raise ValueError(f"ScoreResult at batch index {b} has no progress array")
        p = np.asarray(s.progress, dtype=np.float32)
        if p.shape != (T,):
            raise ValueError(
                f"ScoreResult[{b}].progress has shape {p.shape}, expected ({T},). "
                f"Verify chunk_frames length matches query_steps."
            )
        out[b] = p
    return out


def _require_success_len(scores: List[ScoreResult], T: int) -> np.ndarray:
    """Stack scores[b].success into a (B, T) array; raise if any is missing or off."""
    out = np.empty((len(scores), T), dtype=np.float32)
    for b, s in enumerate(scores):
        if s is None:
            raise ValueError(f"expected ScoreResult at batch index {b}, got None")
        if s.success is None:
            raise ValueError(
                f"ScoreResult[{b}].success is None — the loaded robometer model has no "
                f"success head. Pick a reward_kind that doesn't require success."
            )
        v = np.asarray(s.success, dtype=np.float32)
        if v.shape != (T,):
            raise ValueError(f"ScoreResult[{b}].success has shape {v.shape}, expected ({T},)")
        out[b] = v
    return out


def _post_chunk_shift(scores_BT: np.ndarray) -> np.ndarray:
    """Shift a (B, T) per-chunk-START score array to per-chunk-END alignment.

    chunk_frames[k] is captured BEFORE chunk k runs, so server scores[k] describe
    the pre-chunk-k state. The reward attached to action a_k in the SAC transition
    needs the POST-chunk-k state, which is scores[k+1]. For k = T-1 there is no
    post-chunk frame in our buffer (the rollout ended), so we reuse scores[T-1]
    as a best-effort terminal value — masks[-1] is typically 0 on success anyway,
    so the precise terminal reward rarely affects the bootstrap target.
    """
    out = np.empty_like(scores_BT)
    out[:, :-1] = scores_BT[:, 1:]
    out[:, -1] = scores_BT[:, -1]
    return out


# ----------------------------------------------------------------------------
# Reward functions — each is self-contained and validates its own inputs.
# ----------------------------------------------------------------------------

def _libero_sparse_batched(_scores, metas: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Classic DSRL pi0 reward: -1 per chunk, 0 at terminal if libero succeeded.

    Does NOT use robometer. Input scores are ignored; meta[b]['libero_is_success']
    is the sole signal. Useful as a baseline.
    """
    # scores may be None-list here since we don't use it; only validate metas.
    B = len(metas)
    if B == 0:
        raise ValueError("libero_sparse received empty meta list")
    T = int(metas[0]["query_steps"])
    for b, m in enumerate(metas):
        if int(m["query_steps"]) != T:
            raise ValueError(f"inconsistent query_steps in batch: [0]={T}, [{b}]={m['query_steps']}")
        if "libero_is_success" not in m:
            raise ValueError(f"meta[{b}] missing 'libero_is_success'")
    rewards = np.full((B, T), -1.0, dtype=np.float32)
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        if bool(m["libero_is_success"]):
            rewards[b, -1] = 0.0
            masks[b, -1] = 0.0
    return rewards, masks


def _progress_delta_batched(scores: List[ScoreResult], metas: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """reward[b, t] = progress[b, t+1] - progress[b, t]; last column = 0.

    Naturally aligned to chunk k: the delta IS the change attributable to chunk k
    (post-chunk-k state minus pre-chunk-k state). No `_post_chunk_shift` needed.
    Last column is 0 because there's no post-chunk state for k = T-1.
    """
    B, T = _validate_batch(scores, metas)
    progress = _require_progress_len(scores, T)
    rewards = np.zeros((B, T), dtype=np.float32)
    rewards[:, :-1] = progress[:, 1:] - progress[:, :-1]
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        if bool(m.get("libero_is_success", False)):
            masks[b, -1] = 0.0
    return rewards, masks


def _progress_level_batched(scores: List[ScoreResult], metas: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """reward[b, t] = progress AFTER chunk t (= progress[b, t+1] from server).

    Off-by-one: server progress[k] describes the state BEFORE chunk k. We shift
    so reward[t] credits the outcome of chunk t. See `_post_chunk_shift`.
    """
    B, T = _validate_batch(scores, metas)
    progress = _require_progress_len(scores, T)
    rewards = _post_chunk_shift(progress)
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        if bool(m.get("libero_is_success", False)):
            masks[b, -1] = 0.0
    return rewards, masks


def _success_batched(scores: List[ScoreResult], metas: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """reward[b, t] = success_prob AFTER chunk t (= success[b, t+1] from server).

    Off-by-one: server success[k] describes the state BEFORE chunk k. We shift
    so reward[t] credits the outcome of chunk t. Raises if no success head.
    """
    B, T = _validate_batch(scores, metas)
    success = _require_success_len(scores, T)
    rewards = _post_chunk_shift(success)
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        if bool(m.get("libero_is_success", False)):
            masks[b, -1] = 0.0
    return rewards, masks


def _libero_success_plus_robo_progress_batched(
    scores: List[ScoreResult], metas: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """Libero sparse base + robometer progress AFTER chunk t. Success oracle: libero.

    reward[b, t] = -1 + progress_post[b, t]; terminal -> 0 + progress_post[b, -1]
    on libero success. Off-by-one shift via `_post_chunk_shift`.
    """
    B, T = _validate_batch(scores, metas)
    progress = _require_progress_len(scores, T)
    progress_post = _post_chunk_shift(progress)
    rewards = np.full((B, T), -1.0, dtype=np.float32) + progress_post
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        if bool(m.get("libero_is_success", False)):
            rewards[b, -1] = 0.0 + progress_post[b, -1]
            masks[b, -1] = 0.0
    return rewards, masks


def _robo_success_sparse_batched(scores: List[ScoreResult], metas: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse -1/0 reward, success detected by robometer success head + threshold.

    Requires every ScoreResult to carry a success head. Raises otherwise.
    """
    B, T = _validate_batch(scores, metas)
    success = _require_success_len(scores, T)
    rewards = np.full((B, T), -1.0, dtype=np.float32)
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        threshold = float(m.get("robometer_success_threshold", 0.5))
        if bool(np.any(success[b] >= threshold)):
            rewards[b, -1] = 0.0
            masks[b, -1] = 0.0
    return rewards, masks


def _robo_success_plus_robo_progress_batched(
    scores: List[ScoreResult], metas: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """Sparse base + progress AFTER chunk t, success oracle from robometer success head.

    reward[b, t] = -1 + progress_post[b, t]; terminal -> 0 + progress_post[b, -1]
    when any(success >= threshold). Off-by-one shift via `_post_chunk_shift`.
    The success oracle uses the FULL pre-shift success array — we want to detect
    the success event anywhere in the episode, not just at the post-chunk point.
    Libero is_success is ignored.
    """
    B, T = _validate_batch(scores, metas)
    progress = _require_progress_len(scores, T)
    success = _require_success_len(scores, T)
    progress_post = _post_chunk_shift(progress)
    rewards = np.full((B, T), -1.0, dtype=np.float32) + progress_post
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        threshold = float(m.get("robometer_success_threshold", 0.5))
        if bool(np.any(success[b] >= threshold)):
            rewards[b, -1] = 0.0 + progress_post[b, -1]
            masks[b, -1] = 0.0
    return rewards, masks


REWARD_FNS: Dict[str, RewardFn] = {
    "libero_success_plus_robo_progress": _libero_success_plus_robo_progress_batched,
    "robo_success_plus_robo_progress":   _robo_success_plus_robo_progress_batched,
    "robo_success_sparse":               _robo_success_sparse_batched,
    "progress_delta":                    _progress_delta_batched,
    "progress":                          _progress_level_batched,
    "success":                           _success_batched,
    "libero_sparse":                     _libero_sparse_batched,
}


def get_reward_fn(kind: str) -> RewardFn:
    if kind not in REWARD_FNS:
        raise ValueError(f"unknown robometer_reward_kind={kind!r}; pick from {list(REWARD_FNS)}")
    return REWARD_FNS[kind]


def _resize_video(frames: np.ndarray, frame_size: int) -> np.ndarray:
    """(T, H, W, C) -> (T, frame_size, frame_size, C), BILINEAR; no-op if already right size."""
    if frame_size <= 0 or (frames.shape[1] == frame_size and frames.shape[2] == frame_size):
        return np.ascontiguousarray(frames)
    resized = np.stack(
        [
            np.asarray(PIL.Image.fromarray(f).resize((frame_size, frame_size), PIL.Image.BILINEAR))
            for f in frames
        ],
        axis=0,
    )
    return np.ascontiguousarray(resized)


class PendingScores:
    """Strict-async scoring queue over ``RobometerClient``.

    Scope (deliberately narrow): this class only manages Futures. It does NOT
    apply reward functions — the training loop does that in a single batched
    call. That keeps the "one episode = one request" async contract decoupled
    from "B episodes = one reward computation", and makes all reward logic
    visible at the callsite.

    Lifecycle per episode:
        1. ``submit_episode(traj)``   — fire HTTP request, store (traj, future).
        2. ``drain_ready()``          — pop resolved futures → list of (traj, score).
        3. ``drain_all()``            — block on every remaining future.

    Failure handling is per-future: if a Future raises on ``.result()``:
      * fail_behavior == 'raise'  -> propagate immediately.
      * fail_behavior == 'drop'   -> drop that (traj, future) pair and log stats.
    There is no silent fallback to an alternate reward — if your server breaks,
    you'll either crash or lose the episode, depending on how you configured it.
    """

    def __init__(
        self,
        client: RobometerClient,
        *,
        use_frame_steps: bool = True,
        frame_size: int = 224,
        fail_behavior: str = "raise",
    ) -> None:
        if fail_behavior not in ("raise", "drop"):
            raise ValueError(f"fail_behavior must be 'raise' or 'drop', got {fail_behavior!r}")
        self.client = client
        self.use_frame_steps = use_frame_steps
        self.frame_size = frame_size
        self.fail_behavior = fail_behavior
        self._pending: List[Tuple[Dict, Future]] = []
        self._stats = {"submitted": 0, "ok": 0, "dropped": 0}

    def __len__(self) -> int:
        return len(self._pending)

    def submit_episode(self, traj: Dict, task: str, sample_id: str) -> None:
        frames = _resize_video(traj["chunk_frames"], self.frame_size)
        sample = VideoSample(frames=frames, task=task, id=sample_id)
        future = self.client.submit([sample], use_frame_steps=self.use_frame_steps)
        self._pending.append((traj, future))
        self._stats["submitted"] += 1

    def _resolve_future(self, traj: Dict, future: Future) -> Optional[Tuple[Dict, ScoreResult]]:
        try:
            [score] = future.result()
        except Exception as e:  # pragma: no cover (hits real network)
            if self.fail_behavior == "raise":
                raise
            warnings.warn(f"robometer scoring failed ({e!r}); dropping episode (fail_behavior='drop')")
            self._stats["dropped"] += 1
            return None
        self._stats["ok"] += 1
        return traj, score

    def drain_ready(self) -> List[Tuple[Dict, ScoreResult]]:
        """Pop resolved futures (non-blocking). Drops are silently removed."""
        ready: List[Tuple[Dict, ScoreResult]] = []
        remaining: List[Tuple[Dict, Future]] = []
        for traj, future in self._pending:
            if future.done():
                result = self._resolve_future(traj, future)
                if result is not None:
                    ready.append(result)
            else:
                remaining.append((traj, future))
        self._pending = remaining
        return ready

    def drain_all(self) -> List[Tuple[Dict, ScoreResult]]:
        """Block on every remaining future. Drops are silently removed."""
        out: List[Tuple[Dict, ScoreResult]] = []
        for traj, future in self._pending:
            result = self._resolve_future(traj, future)
            if result is not None:
                out.append(result)
        self._pending = []
        return out

    def popleft_oldest(self) -> Optional[Tuple[Dict, ScoreResult]]:
        """Pop+resolve the oldest future, blocking until it's done. None if empty."""
        if not self._pending:
            return None
        traj, future = self._pending.pop(0)
        return self._resolve_future(traj, future)

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)


def _build_meta(traj: Dict, robometer_success_threshold: float) -> Dict:
    """Per-episode meta consumed by reward functions."""
    return {
        "query_steps": len(traj["actions"]),
        "libero_is_success": bool(traj["is_success"]),
        "robometer_success_threshold": float(robometer_success_threshold),
    }


def _apply_reward_fn(
    reward_fn: RewardFn,
    resolved: List[Tuple[Dict, ScoreResult]],
    robometer_success_threshold: float,
) -> List[Tuple[Dict, np.ndarray, np.ndarray, ScoreResult]]:
    """Run the batched reward fn on all resolved episodes and split per-episode."""
    if not resolved:
        return []
    scores = [s for _, s in resolved]
    metas = [_build_meta(t, robometer_success_threshold) for t, _ in resolved]
    rewards_B, masks_B = reward_fn(scores, metas)
    rewards_B = np.asarray(rewards_B, dtype=np.float32)
    masks_B = np.asarray(masks_B, dtype=np.float32)
    B = len(resolved)
    T = metas[0]["query_steps"]
    if rewards_B.shape != (B, T) or masks_B.shape != (B, T):
        raise ValueError(
            f"reward_fn returned rewards.shape={rewards_B.shape}, masks.shape={masks_B.shape}; "
            f"expected {(B, T)} for both."
        )
    return [
        (traj, rewards_B[b], masks_B[b], score)
        for b, (traj, score) in enumerate(resolved)
    ]

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    # DSRL-NA batches carry non-jit-able fields (`original_observations` is a list
    # of dicts containing string prompts; same for `original_next_observations`).
    # The DSRLNA buffer's iterator already device-puts the jit-able fields, so we
    # skip the outer shard_fn wrap in NA mode.
    _na_skip_shard = getattr(variant, 'algorithm', 'pixel_sac') == 'pixel_dsrl_na'
    if shard_fn is not None and not _na_skip_shard:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    episode_count = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    # ------------------------------------------------------------------
    # Robometer async scoring. Strict policy: every episode's reward goes
    # through client.submit(...), never client.score(...). Rollouts for
    # episode N+1 and SAC updates proceed while episode N is still being
    # scored on the server.
    # ------------------------------------------------------------------
    reward_fn = get_reward_fn(variant.robometer_reward_kind)
    with RobometerClient(
        variant.robometer_url,
        max_concurrent_requests=max(8, variant.robometer_queue_max_depth + 2),
        max_retries=variant.robometer_max_retries,
    ) as client:
        pending = PendingScores(
            client,
            use_frame_steps=bool(variant.robometer_use_frame_steps),
            frame_size=variant.robometer_frame_size,
            fail_behavior=variant.robometer_fail_behavior,
        )
        print("robometer health:", client.health())
        # Surface the SAC action shape so historical wandb runs can be filtered
        # "lifted vs unlifted" (see docs/sac_action_lift_plan.md).
        print(
            f"sac action_chunk_shape = {tuple(agent.action_chunk_shape)} "
            f"(action_dim_total = {int(agent.action_dim)})"
        )
        wandb_logger.log({
            'sac/action_chunk_len': int(agent.action_chunk_shape[0]),
            'sac/action_latent_dim': int(agent.action_chunk_shape[1]),
            'sac/action_dim_total': int(agent.action_dim),
        }, step=0)

        algorithm = getattr(variant, 'algorithm', 'pixel_sac')
        is_na = algorithm == 'pixel_dsrl_na'
        scattered_mode = str(getattr(variant, 'scattered_mode', 'off'))
        is_scattered = scattered_mode != 'off'
        if is_scattered:
            if is_na:
                raise ValueError(
                    "scattered_mode != 'off' is not supported with pixel_dsrl_na yet "
                    "(see docs/scattered_training_samples_plan.md, Phase 3)."
                )
            if int(variant.sac_action_chunk_size) != int(variant.query_freq):
                raise ValueError(
                    f"scattered_mode={scattered_mode!r} requires lifted action chunks "
                    f"(sac_action_chunk_size == query_freq); got "
                    f"sac_action_chunk_size={variant.sac_action_chunk_size}, query_freq={variant.query_freq}."
                )

        def _insert_batched(resolved_pairs):
            """Apply reward_fn to a batch of resolved (traj, score) pairs and insert."""
            nonlocal total_env_steps
            if not resolved_pairs:
                return
            scored = _apply_reward_fn(
                reward_fn, resolved_pairs, variant.robometer_success_threshold
            )
            for traj_done, rewards, masks, score in scored:
                if is_na:
                    _insert_scored_traj_na(variant, traj_done, rewards, masks, score,
                                           online_replay_buffer, wandb_logger, i)
                else:
                    _insert_scored_traj(variant, traj_done, rewards, masks, score,
                                        online_replay_buffer, wandb_logger, i)
                total_env_steps += traj_done['env_steps']

        with tqdm(total=variant.max_steps, initial=0) as pbar:
            while i <= variant.max_steps:
                # 1) Collect a new rollout (blocking physics sim).
                traj = collect_traj(variant, agent, env, i, agent_dp)
                # Flush rollout-phase events into the trace JSONL so the
                # chrome viewer shows the rollout band between SAC outer steps.
                _na_flush_rollout(variant)
                if is_scattered:
                    # Scattered mode short-circuits the chunk-grid HTTP pipeline:
                    # the mock scorer is in-process, and one episode's samples
                    # are emitted synchronously into the buffer.
                    total_env_steps += _score_and_insert_scattered(
                        variant, traj, online_replay_buffer, wandb_logger, i
                    )
                    episode_count += 1
                else:
                    # 2) Fire-and-forget scoring for this episode.
                    pending.submit_episode(
                        traj,
                        task=str(variant.task_description),
                        sample_id=f"ep_{episode_count}",
                    )
                    episode_count += 1

                    # 3) If the queue grew past the configured max depth, block on the
                    #    oldest one before the next rollout — otherwise memory piles up.
                    over_depth_batch = []
                    while len(pending) > variant.robometer_queue_max_depth:
                        popped = pending.popleft_oldest()
                        if popped is not None:
                            over_depth_batch.append(popped)
                    _insert_batched(over_depth_batch)

                    # 4) Drain any futures that already resolved while physics ran, and
                    #    score the whole set in ONE reward_fn call (batched).
                    _insert_batched(pending.drain_ready())

                # 5) If the buffer is still not warm enough to train, block on the
                #    remaining queue — we can't proceed without data anyway.
                # Reference DSRL-NA gates by *trajectory* count
                # (`num_initial_traj_collect`), so honour that knob if set;
                # otherwise fall back to our buffer-timestep threshold
                # (`start_online_updates`).
                init_trajs = int(getattr(variant, 'num_initial_traj_collect', -1))
                if init_trajs > 0:
                    not_warm = online_replay_buffer._traj_counter < init_trajs
                else:
                    not_warm = len(online_replay_buffer) <= variant.start_online_updates
                if not_warm and len(pending) > 0:
                    _insert_batched(pending.drain_all())

                traj_id = online_replay_buffer._traj_counter
                tqdm.write(
                    f'online buffer timesteps length: {len(online_replay_buffer)} | '
                    f'num traj: {traj_id} | total env steps: {total_env_steps} | '
                    f'robometer queue depth: {len(pending)} stats: {pending.stats()}'
                )

                if variant.get("num_online_gradsteps_batch", -1) > 0:
                    num_gradsteps = variant.num_online_gradsteps_batch
                else:
                    num_gradsteps = len(traj["actions"]) * variant.multi_grad_step

                # Same gate semantics as the warm-up drain above.
                if init_trajs > 0:
                    can_train = online_replay_buffer._traj_counter >= init_trajs
                else:
                    can_train = len(online_replay_buffer) > variant.start_online_updates
                if can_train:
                    for _ in range(num_gradsteps):
                        if i == 0:
                            print('performing evaluation for initial checkpoint')
                            if perform_control_evals:
                                perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                            if hasattr(agent, 'perform_eval'):
                                agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                        batch = next(replay_buffer_iterator)
                        if is_na:
                            update_info = _na_update_step(
                                agent, agent_dp, variant, batch, online_replay_buffer,
                                step_idx=i,
                            )
                        else:
                            update_info = agent.update(batch)

                        pbar.update()
                        i += 1

                        if i % variant.log_interval == 0:
                            update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                            for k, v in update_info.items():
                                v_arr = np.asarray(v)
                                if v_arr.ndim == 0:
                                    wandb_logger.log({f'training/{k}': v_arr}, step=i)
                                elif v_arr.ndim <= 2:
                                    wandb_logger.log_histogram(f'training/{k}', v_arr, i)
                            wandb_logger.log({
                                'replay_buffer_size': len(online_replay_buffer),
                                'episode_return (exploration)': traj['episode_return'],
                                'is_success (exploration)': int(traj['is_success']),
                                'robometer/queue_depth': len(pending),
                                'robometer/submitted': pending.stats()['submitted'],
                                'robometer/ok': pending.stats()['ok'],
                                'robometer/dropped': pending.stats()['dropped'],
                            }, i)

                        if i % variant.eval_interval == 0:
                            wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                            wandb_logger.log({'num_online_trajs': traj_id}, step=i)
                            wandb_logger.log({'env_steps': total_env_steps}, step=i)
                            if perform_control_evals:
                                perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                            if hasattr(agent, 'perform_eval'):
                                agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                        if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                            agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)


def _insert_scored_traj_na(variant, traj, rewards, masks, scores, online_replay_buffer,
                           wandb_logger, step):
    """DSRL-NA insert path: delegates to the reference add_online_data_to_buffer_na
    after injecting the rewards/masks computed by our robometer reward_fn pipeline.
    """
    from examples.train_utils_robometer_na import add_online_data_to_buffer_na
    traj_with_rewards = dict(traj)
    traj_with_rewards['rewards'] = np.asarray(rewards, dtype=np.float32)
    traj_with_rewards['masks'] = np.asarray(masks, dtype=np.float32)
    add_online_data_to_buffer_na(variant, traj_with_rewards, online_replay_buffer)
    log_payload = {
        'episode_return (libero)': traj['episode_return'],
        'is_success (libero)': int(traj['is_success']),
        'robometer/reward_sum': float(np.sum(rewards)),
    }
    if scores is not None:
        progress = np.asarray(scores.progress, dtype=np.float32)
        log_payload['robometer/final_progress'] = float(progress[-1]) if len(progress) else 0.0
        log_payload['robometer/mean_progress'] = float(progress.mean()) if len(progress) else 0.0
        if scores.success is not None and len(scores.success):
            log_payload['robometer/final_success'] = float(np.asarray(scores.success)[-1])
    wandb_logger.log(log_payload, step=step)


_NA_TRACE_T0 = None
_NA_CHROME_FILE = None  # open file handle for the streaming chrome JSON trace
_NA_CHROME_NEEDS_COMMA = False  # True after first event written

# Process and per-component thread tags surfaced in chrome://tracing.
_NA_CHROME_PID = 'dsrl_na'
_NA_CHROME_TIDS = {
    # Rollout phase (env step + pi0 inference).
    'rollout_kv_extract': 'rollout',
    'rollout_pi0_infer': 'rollout',
    # SAC update phase, per-component.
    'kv_fetch': 'kv_fetch',
    'ac_pi0': 'action_critic_pi0',
    'ac_update': 'action_critic_update',
    'nc_pi0': 'noise_critic_pi0',
    'nc_update': 'noise_critic_update',
    'na_update': 'noise_actor_update',
    'outer_step': 'outer_sac',
}


def _open_chrome_trace_file(variant):
    """Open the chrome JSON trace file lazily and write the header. Idempotent."""
    import os as _os
    import atexit as _atexit
    import json as _json
    global _NA_CHROME_FILE, _NA_CHROME_NEEDS_COMMA
    if _NA_CHROME_FILE is not None:
        return _NA_CHROME_FILE
    path = _na_trace_path(variant)
    if path is None:
        return None
    _os.makedirs(_os.path.dirname(path) or '.', exist_ok=True)
    fh = open(path, 'w')
    fh.write('{"traceEvents":[\n')
    # Process metadata so chrome shows a friendly name.
    fh.write(_json.dumps({
        'name': 'process_name', 'ph': 'M', 'pid': _NA_CHROME_PID, 'tid': 0,
        'args': {'name': _NA_CHROME_PID},
    }))
    _NA_CHROME_NEEDS_COMMA = True
    fh.flush()
    _NA_CHROME_FILE = fh

    def _close():
        global _NA_CHROME_FILE
        try:
            if _NA_CHROME_FILE is not None:
                _NA_CHROME_FILE.write('\n],"displayTimeUnit":"ms"}\n')
                _NA_CHROME_FILE.close()
        except Exception:
            pass
        _NA_CHROME_FILE = None
    _atexit.register(_close)
    return _NA_CHROME_FILE


def _emit_chrome(variant, name, t_s, dur_s, tid, args=None):
    """Write one X (complete) event in chrome trace format, microsecond units."""
    import json as _json
    global _NA_CHROME_NEEDS_COMMA
    fh = _open_chrome_trace_file(variant)
    if fh is None or dur_s <= 0:
        return
    event = {
        'name': name, 'ph': 'X',
        'ts': int(round(t_s * 1e6)),
        'dur': int(round(dur_s * 1e6)),
        'pid': _NA_CHROME_PID,
        'tid': tid,
    }
    if args:
        event['args'] = args
    sep = ',\n' if _NA_CHROME_NEEDS_COMMA else '\n'
    fh.write(sep + _json.dumps(event))
    _NA_CHROME_NEEDS_COMMA = True


def _na_rollout_event(variant, name, t_start, dur, **extra):
    """Emit a rollout-phase event directly into the chrome trace."""
    tid = _NA_CHROME_TIDS.get(name, name)
    _emit_chrome(variant, name, t_start, dur, tid, extra or None)


def _na_flush_rollout(variant):
    """No-op now: events are streamed directly in `_na_rollout_event`. Kept
    for call-site compatibility (called once per trajectory).
    """
    fh = _open_chrome_trace_file(variant)
    if fh is not None:
        fh.flush()


def _na_now():
    """Seconds since the trace anchor (lazy init on first call)."""
    import time as _time
    global _NA_TRACE_T0
    if _NA_TRACE_T0 is None:
        _NA_TRACE_T0 = _time.perf_counter()
    return _time.perf_counter() - _NA_TRACE_T0


def _na_trace_path(variant):
    """Resolve the JSONL trace file path for per-component update timings.

    Order of precedence:
      1. ``$DSRL_NA_TRACE`` env var (explicit override).
      2. ``variant.na_trace_file`` if set.
      3. Default: ``./traces/na_trace_<timestamp>.jsonl`` in the current
         working directory (the repo root when launched from the project).
    Returns ``None`` if tracing is disabled (``DSRL_NA_TRACE='off'``).
    """
    import os as _os
    import time as _time
    env_path = _os.environ.get('DSRL_NA_TRACE')
    if env_path == 'off':
        return None
    if env_path:
        return env_path
    cli_path = getattr(variant, 'na_trace_file', '') or ''
    if cli_path:
        return cli_path
    # Default: ./traces/ in CWD with a stable per-run timestamp so reruns
    # don't append onto each other.
    if not hasattr(_na_trace_path, '_default_path'):
        ts = _time.strftime('%Y%m%d_%H%M%S')
        _na_trace_path._default_path = _os.path.join('traces', f'na_chrome_{ts}.json')
    return _na_trace_path._default_path


def _block_until_ready(tree):
    """Force a JAX tree to be fully evaluated on its current device. Required
    for accurate timing — JAX dispatches ops asynchronously, so wall-clock
    around an unsynchronized op only captures Python+dispatch overhead.
    """
    try:
        jax.block_until_ready(tree)
    except Exception:
        # Fallback: walk the tree and call .block_until_ready() on jax arrays.
        def _b(x):
            if hasattr(x, 'block_until_ready'):
                x.block_until_ready()
            return x
        jax.tree_util.tree_map(_b, tree)


def _na_update_step(agent, agent_dp, variant, batch, online_replay_buffer, step_idx=None):
    """Wrap one outer DSRL-NA SAC step. Replicates the arayabrain reference's
    nested-step structure when ``train_all_together=0`` semantics are
    requested via ``--action_critic_steps`` / ``--noise_critic_steps`` /
    ``--noise_actor_steps``:

    - For each of ``action_critic_steps`` (default 1): sample next-noise,
      run pi0 on next_obs to build ``next_executed_actions``, run an
      action-critic update.
    - For each of ``noise_critic_steps`` (default 1): generate a fresh
      ``(distill_noise, distill_actions)`` pair via pi0 forward, run a
      noise-critic distillation update.
    - For each of ``noise_actor_steps`` (default 1): run an actor update
      against the (already-trained) noise critic; pi0 is **not** invoked.

    The toggle flags ``--train_action_critic`` / ``--train_noise_actor``
    / ``--train_noise_critic`` short-circuit each whole inner-loop block.

    Per-component timing is written to ``DSRL_NA_TRACE`` (JSONL, one record
    per outer step). Each record breaks out:
      - ``kv_fetch_s``: total time to page K/V (current + next obs) to GPU.
      - ``ac_pi0_s`` / ``ac_update_s``: list of per-iter pi0 forward + jitted
        update times for the action-critic block.
      - ``nc_pi0_s`` / ``nc_update_s``: same for noise-critic distillation.
      - ``na_update_s``: list of jitted-update times for the actor block
        (no pi0 forwards by design).
    Each timed region is bracketed by ``jax.block_until_ready(...)`` so the
    measurement reflects actual GPU work, not async-dispatch overhead.
    """
    import time as _time
    from examples.train_utils_robometer_na import (
        get_next_actions_from_dp, generate_distillation_batch, remove_original_obs_keys,
    )
    from flax.core import frozen_dict as _fd

    robot_config = {
        'action_chunk_size': 50,           # pi0 libero internal horizon
        'use_local_policy': True,          # we run pi0 locally (vs WebsocketClient)
        'save_kv_cache': bool(int(getattr(variant, 'save_kv_cache', 0))),
        'max_timesteps': int(getattr(variant, 'max_timesteps', 400)),
    }
    train_action_critic = bool(int(getattr(variant, 'train_action_critic', 1)))
    train_noise_actor = bool(int(getattr(variant, 'train_noise_actor', 1)))
    train_noise_critic = bool(int(getattr(variant, 'train_noise_critic', 1)))
    n_ac = int(getattr(variant, 'action_critic_steps', 1)) if train_action_critic else 0
    n_nc = int(getattr(variant, 'noise_critic_steps', 1)) if train_noise_critic else 0
    n_na = int(getattr(variant, 'noise_actor_steps', 1)) if train_noise_actor else 0
    save_kv = robot_config['save_kv_cache']

    base_batch = remove_original_obs_keys(batch)
    info = {}

    # Streaming chrome trace: each timed region is emitted as one X-phase
    # event with absolute start offset (seconds since the trace anchor).
    # Bubbles between outer steps surface naturally because we never write
    # intermediate offsets.
    step_for_args = int(step_idx) if step_idx is not None else -1
    outer_start = _na_now()
    t_outer = _time.perf_counter()

    # Pre-fetch K/V slices once per outer step (same indices reused across
    # inner iterations on the same batch). When put_kv_cache_on_cpu=1 the
    # cache lives in host memory; pi0's jit expects all inputs on the same
    # GPU as its weights, so we page back to the agent_dp device here
    # (mirrors the reference's ``agent_dp_device_proxy`` paging).
    next_kv = None
    cur_kv = None
    if save_kv:
        try:
            agent_dp_device = jax.devices('gpu')[0]
        except RuntimeError:
            agent_dp_device = jax.devices()[0]

        def _page_to_gpu(arr_list):
            paged = [jax.device_put(a, agent_dp_device) for a in arr_list]
            _block_until_ready(paged)
            return paged

        t_kv_start = _na_now()
        t_kv = _time.perf_counter()
        if 'next_indices' in batch:
            next_kv = (
                _page_to_gpu(online_replay_buffer.get_cache(batch['next_indices'], 'k')),
                _page_to_gpu(online_replay_buffer.get_cache(batch['next_indices'], 'v')),
            )
        if 'indices' in batch:
            cur_kv = (
                _page_to_gpu(online_replay_buffer.get_cache(batch['indices'], 'k')),
                _page_to_gpu(online_replay_buffer.get_cache(batch['indices'], 'v')),
            )
        _emit_chrome(variant, 'kv_fetch', t_kv_start, _time.perf_counter() - t_kv,
                     _NA_CHROME_TIDS['kv_fetch'], {'step': step_for_args})

    # ---- Inner action-critic loop ----------------------------------------
    ac_pi0_total = 0.0
    ac_update_total = 0.0
    for j in range(n_ac):
        t_start = _na_now()
        t_pi0 = _time.perf_counter()
        next_noise, next_log_probs = agent.sample_actions_with_log_probs(batch['next_observations'])
        batch_dict = dict(batch)
        if save_kv and next_kv is not None:
            batch_dict['original_next_k_cache'] = next_kv[0]
            batch_dict['original_next_v_cache'] = next_kv[1]
        agent_dp_next, _times = get_next_actions_from_dp(
            agent_dp, batch_dict, next_noise, agent.action_chunk_shape, robot_config, variant,
        )
        _block_until_ready((next_noise, next_log_probs, agent_dp_next))
        d = _time.perf_counter() - t_pi0
        ac_pi0_total += d
        _emit_chrome(variant, 'ac_pi0', t_start, d,
                     _NA_CHROME_TIDS['ac_pi0'], {'step': step_for_args, 'inner': j})

        update_batch = dict(base_batch)
        update_batch['next_executed_actions'] = agent_dp_next
        update_batch['next_log_probs'] = next_log_probs
        t_start = _na_now()
        t_upd = _time.perf_counter()
        out = agent.update(
            _fd.freeze(update_batch), distill_batch=None,
            train_action_critic=True, train_noise_actor=False,
        )
        _block_until_ready(out)
        d = _time.perf_counter() - t_upd
        ac_update_total += d
        _emit_chrome(variant, 'ac_update', t_start, d,
                     _NA_CHROME_TIDS['ac_update'], {'step': step_for_args, 'inner': j})
        info.update(out)

    # ---- Inner noise-critic distillation loop ----------------------------
    nc_pi0_total = 0.0
    nc_update_total = 0.0
    for j in range(n_nc):
        t_start = _na_now()
        t_pi0 = _time.perf_counter()
        distill_input = dict(batch)
        if save_kv and cur_kv is not None:
            distill_input['original_k_cache'] = cur_kv[0]
            distill_input['original_v_cache'] = cur_kv[1]
        distill_noise, distill_actions, _times = generate_distillation_batch(
            distill_input, agent, agent_dp, robot_config, variant,
        )
        _block_until_ready((distill_noise, distill_actions))
        d = _time.perf_counter() - t_pi0
        nc_pi0_total += d
        _emit_chrome(variant, 'nc_pi0', t_start, d,
                     _NA_CHROME_TIDS['nc_pi0'], {'step': step_for_args, 'inner': j})

        distill_batch = _fd.freeze({
            'distill_noise': distill_noise,
            'distill_actions': distill_actions,
        })
        t_start = _na_now()
        t_upd = _time.perf_counter()
        out = agent.update(
            _fd.freeze(base_batch), distill_batch=distill_batch,
            train_action_critic=False, train_noise_actor=False,
        )
        _block_until_ready(out)
        d = _time.perf_counter() - t_upd
        nc_update_total += d
        _emit_chrome(variant, 'nc_update', t_start, d,
                     _NA_CHROME_TIDS['nc_update'], {'step': step_for_args, 'inner': j})
        info.update(out)

    # ---- Inner noise-actor loop (no pi0) ---------------------------------
    na_update_total = 0.0
    for j in range(n_na):
        t_start = _na_now()
        t_upd = _time.perf_counter()
        out = agent.update(
            _fd.freeze(base_batch), distill_batch=None,
            train_action_critic=False, train_noise_actor=True,
        )
        _block_until_ready(out)
        d = _time.perf_counter() - t_upd
        na_update_total += d
        _emit_chrome(variant, 'na_update', t_start, d,
                     _NA_CHROME_TIDS['na_update'], {'step': step_for_args, 'inner': j})
        info.update(out)

    outer_d = _time.perf_counter() - t_outer
    _emit_chrome(variant, 'outer_step', outer_start, outer_d,
                 _NA_CHROME_TIDS['outer_step'], {'step': step_for_args})

    # Aggregate scalars surfaced to wandb (means across the inner loops).
    info['na/timing/total_s'] = outer_d
    info['na/timing/ac_pi0_mean_s'] = ac_pi0_total / n_ac if n_ac else 0.0
    info['na/timing/ac_update_mean_s'] = ac_update_total / n_ac if n_ac else 0.0
    info['na/timing/nc_pi0_mean_s'] = nc_pi0_total / n_nc if n_nc else 0.0
    info['na/timing/nc_update_mean_s'] = nc_update_total / n_nc if n_nc else 0.0
    info['na/timing/na_update_mean_s'] = na_update_total / n_na if n_na else 0.0
    info['na/inner_action_critic_steps'] = n_ac
    info['na/inner_noise_critic_steps'] = n_nc
    info['na/inner_noise_actor_steps'] = n_na
    return info


def _score_and_insert_scattered(variant, traj, online_replay_buffer, wandb_logger, step):
    """Score one episode with the scattered-frame robometer and insert.

    Synchronous (no async queue) because the mock is in-process; when the real
    scattered-robometer endpoint exists we'll move this to a Future-backed
    pipeline mirroring ``PendingScores``.

    Returns the number of env steps consumed (for total_env_steps accounting).
    """
    from examples.scattered_robometer import MockScatteredScorer
    from jaxrl2.data import scattered_samples as ss

    H = int(variant.query_freq)
    mode = str(variant.scattered_mode)
    T = int(traj['env_steps'])

    obs_stream = traj['scattered_obs_stream']
    noise_stream = traj['scattered_noise_stream']
    executed_stream = traj['scattered_executed_stream']
    libero_rewards = np.asarray(traj['libero_rewards_per_step'], dtype=np.float32)[:T]
    libero_success_step = traj['libero_success_step']

    if not bool(int(getattr(variant, 'scattered_use_mock', 1))):
        raise NotImplementedError(
            "scattered_use_mock=0 selected, but no real scattered-frame robometer "
            "endpoint is wired up yet."
        )

    scorer = MockScatteredScorer(
        num_selected=int(variant.scattered_num_selected),
        strategy=str(variant.scattered_strategy),
        progress_oracle=str(variant.scattered_progress_oracle),
        emit_success=bool(int(variant.scattered_emit_success)),
        seed=int(variant.scattered_seed),
    )
    # The mock doesn't read pixel content; pass a length-T stub so the API
    # contract (4D video) is satisfied for forward-compat with the real server.
    video_stub = np.zeros((T, 1, 1, 3), dtype=np.uint8)
    sample_id = f"ep_{online_replay_buffer._traj_counter}"
    score = scorer.score(
        video=video_stub,
        task=str(variant.task_description),
        sample_id=sample_id,
        libero_rewards=libero_rewards,
        libero_success_step=libero_success_step,
    )

    progress_dense = ss.interpolate_progress(score.selected_indices, score.progress, T)
    if score.success is not None:
        success_dense = ss.interpolate_success(score.selected_indices, score.success, T)
    else:
        success_dense = np.zeros((T,), dtype=np.float32)

    rng = np.random.default_rng(int(variant.scattered_seed) + online_replay_buffer._traj_counter)
    if mode == 'look_future':
        samples = ss.construct_look_future(
            selected_indices=score.selected_indices,
            obs_stream=obs_stream,
            noise_stream=noise_stream,
            executed_stream=executed_stream,
            progress_dense=progress_dense,
            success_dense=success_dense,
            H=H,
            env=variant.env,
        )
    elif mode == 'look_history':
        samples = ss.construct_look_history(
            selected_indices=score.selected_indices,
            obs_stream=obs_stream,
            noise_stream=noise_stream,
            executed_stream=executed_stream,
            progress_dense=progress_dense,
            success_dense=success_dense,
            H=H,
            env=variant.env,
        )
    elif mode == 'random_subsample':
        samples = ss.construct_random_subsample(
            obs_stream=obs_stream,
            noise_stream=noise_stream,
            executed_stream=executed_stream,
            progress_dense=progress_dense,
            success_dense=success_dense,
            H=H,
            env=variant.env,
            n=int(variant.scattered_random_n),
            rng=rng,
        )
    else:
        raise ValueError(f"unknown scattered_mode={mode!r}")

    if not samples:
        warnings.warn(f"scattered mode={mode!r} produced 0 samples for episode {sample_id}")
        return int(traj['env_steps'])

    reward_fn = get_reward_fn(variant.robometer_reward_kind)
    rewards, masks = ss.apply_reward_fn_scattered(
        reward_fn,
        samples,
        libero_is_success=bool(traj['is_success']),
        libero_success_step=libero_success_step,
        H=H,
        robometer_success_threshold=float(variant.robometer_success_threshold),
        emit_success=score.success is not None,
        reward_kind=str(variant.robometer_reward_kind),
    )

    expected_action_shape = tuple(online_replay_buffer.action_space.shape)
    for k, sample in enumerate(samples):
        next_sample = samples[k + 1] if k < len(samples) - 1 else sample
        obs = {kk: vv[0] for kk, vv in sample.obs.items()}
        next_obs = {kk: vv[0] for kk, vv in sample.next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        if tuple(sample.noise.shape) != expected_action_shape:
            raise ValueError(
                f"scattered sample noise shape {sample.noise.shape} != buffer action_space {expected_action_shape}"
            )
        online_replay_buffer.insert(dict(
            observations=obs,
            next_observations=next_obs,
            actions=sample.noise,
            next_actions=next_sample.noise,
            rewards=float(rewards[k]),
            masks=float(masks[k]),
            discount=variant.discount ** H,
        ))
    online_replay_buffer.increment_traj_counter()

    log_payload = {
        'episode_return (libero)': traj['episode_return'],
        'is_success (libero)': int(traj['is_success']),
        'scattered/num_samples': len(samples),
        'scattered/num_padded': int(sum(1 for s in samples if s.padded)),
        'scattered/reward_sum': float(np.sum(rewards)),
        'scattered/mean_reward': float(np.mean(rewards)),
    }
    if len(score.progress):
        log_payload['scattered/final_progress'] = float(score.progress[-1])
        log_payload['scattered/mean_progress'] = float(score.progress.mean())
    if score.success is not None and len(score.success):
        log_payload['scattered/final_success'] = float(score.success[-1])
    wandb_logger.log(log_payload, step=step)
    return int(traj['env_steps'])


def _insert_scored_traj(variant, traj, rewards, masks, scores, online_replay_buffer, wandb_logger, step):
    """Insert one scored episode into the replay buffer and log robometer stats."""
    add_online_data_to_buffer(variant, traj, rewards, masks, online_replay_buffer)
    log_payload = {
        'episode_return (libero)': traj['episode_return'],
        'is_success (libero)': int(traj['is_success']),
        'robometer/reward_sum': float(np.sum(rewards)),
    }
    if scores is not None:
        progress = np.asarray(scores.progress, dtype=np.float32)
        log_payload['robometer/final_progress'] = float(progress[-1]) if len(progress) else 0.0
        log_payload['robometer/mean_progress'] = float(progress.mean()) if len(progress) else 0.0
        if scores.success is not None and len(scores.success):
            log_payload['robometer/final_success'] = float(np.asarray(scores.success)[-1])
    wandb_logger.log(log_payload, step=step)

            
def add_online_data_to_buffer(variant, traj, rewards, masks, online_replay_buffer):
    """Insert one scored episode into the SAC replay buffer.

    Unlike ``train_utils_sim.add_online_data_to_buffer``, the reward/mask vectors
    are passed in (computed from robometer scores upstream) rather than read from
    the traj dict — collect_traj no longer rewrites them.
    """
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim)  e.g. (20, 20, 32) post-lift
    episode_len = len(actions)
    rewards = np.asarray(rewards, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    assert len(rewards) == episode_len, f"rewards len {len(rewards)} != {episode_len}"
    assert len(masks) == episode_len, f"masks len {len(masks)} != {episode_len}"
    # Catch shape regressions early — buffer is initialized once at startup with
    # the action shape from DummyEnv.action_space, so a mismatch here means
    # something upstream (DummyEnv or PixelSACLearner) drifted out of sync.
    expected_action_shape = tuple(online_replay_buffer.action_space.shape)
    assert tuple(actions.shape[1:]) == expected_action_shape, (
        f"chunk action shape mismatch: got {actions.shape[1:]}, "
        f"buffer expects {expected_action_shape}"
    )

    # DSRL-NA path is handled separately by add_online_data_to_buffer_na
    # (in examples/train_utils_robometer_na.py). This function is the
    # DSRL-SAC insert path only.
    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward

    agent._rng, rng = jax.random.split(agent._rng)

    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()

    image_list = [] # for visualization
    rewards = []
    action_list = []                           # SAC noise latents per chunk: (T_act, D_noise=32)
    diffused_action_list: List[np.ndarray] = []  # pi0 env-actions per chunk: (T_act, D_act=7 for libero)
    obs_list = []
    chunk_frames: List[np.ndarray] = []   # 256x256 agentview frames at chunk boundaries
    chunk_env_steps: List[int] = []

    # Scattered training samples: per-env-step streams, populated only when
    # variant.scattered_mode != 'off'. Zero overhead in the default path.
    scattered_mode = getattr(variant, 'scattered_mode', 'off')
    record_scattered = scattered_mode != 'off'
    scattered_obs_stream: List[dict] = []
    scattered_executed_stream: List[np.ndarray] = []
    scattered_noise_chunks: List[np.ndarray] = []  # one (query_freq, noise_dim) row per chunk

    # DSRL-NA bookkeeping (no-op for DSRL-SAC). The pi0 prompt-encoded inputs are
    # stored verbatim per chunk so the per-update distillation can re-run pi0
    # against the same observation cheaply via cached K/V.
    is_na = getattr(variant, 'algorithm', 'pixel_sac') == 'pixel_dsrl_na'
    save_kv_cache = is_na and bool(int(getattr(variant, 'save_kv_cache', 0)))
    original_obs_list: List[dict] = []
    k_cache_outs: List = []
    v_cache_outs: List = []

    # Always run to the full horizon — robometer needs to see the whole episode,
    # not an early-exit stub. The libero success step is still recorded below.
    libero_success_step: Optional[int] = None

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)

        qpos = obs_to_qpos(obs, variant)

        if variant.add_states:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
                'state': qpos[np.newaxis, ..., np.newaxis],
            }
        else:
            obs_dict = {
                'pixels': curr_image[np.newaxis, ..., np.newaxis],
            }

        if t % query_frequency == 0:

            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)

            # Optional pi0 K/V cache extraction for DSRL-NA — lets the per-update
            # distillation reuse the prompt-encoder K/V instead of re-running it.
            kv_cache = None
            if is_na and save_kv_cache:
                _t = _na_now()
                _t_perf = time.perf_counter()
                rep_out = agent_dp.get_prefix_rep_and_kv_cache(obs_pi_zero)
                kv_cache = rep_out['kv_cache']
                k, v = kv_cache
                # Optional CPU offload (matches reference's put_kv_cache_on_cpu=1):
                # at large buffer capacity (e.g. 150_000) the K/V on GPU exhausts
                # memory; pinning to host RAM and paging back at update time is
                # the standard fix.
                if bool(int(getattr(variant, 'put_kv_cache_on_cpu', 0))):
                    cpu_dev = jax.devices('cpu')[0]
                    k = jax.device_put(k, cpu_dev)
                    v = jax.device_put(v, cpu_dev)
                _block_until_ready((k, v))
                _na_rollout_event(variant, 'rollout_kv_extract', _t, time.perf_counter() - _t_perf, env_step=int(t))
                k_cache_outs.append(k)
                v_cache_outs.append(v)

            if i == 0:
                # for initial round of data collection, we sample from standard gaussian noise
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                # sac agent predicts the noise for diffusion model
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]

            # Pass the cached K/V to pi0 in NA+save_kv_cache mode. agent_dp.infer
            # tolerates kv_cache=None as the no-cache path.
            infer_kwargs = {'noise': noise}
            if kv_cache is not None:
                infer_kwargs['kv_cache'] = kv_cache
            _t = _na_now()
            _t_perf = time.perf_counter()
            actions = agent_dp.infer(obs_pi_zero, **infer_kwargs)["actions"]
            _block_until_ready(actions)
            _na_rollout_event(variant, 'rollout_pi0_infer', _t, time.perf_counter() - _t_perf,
                              env_step=int(t), cached_kv=int(kv_cache is not None))
            action_list.append(actions_noise)
            # Diffused env-actions for the chunk — first ``query_frequency`` of
            # pi0's 50-step output are the ones actually executed. Stored for
            # DSRL-NA's diffused-action critic; ignored by DSRL-SAC.
            diffused_action_list.append(np.asarray(actions[:query_frequency], dtype=np.float32))
            obs_list.append(obs_dict)
            if is_na:
                # Deepcopy the pi0-format observation so each chunk's stored entry
                # doesn't alias future mutations of obs_pi_zero.
                from copy import deepcopy as _deepcopy
                original_obs_list.append(_deepcopy(obs_pi_zero))
            # 256x256 chunk-boundary frame for robometer. Libero's agentview is
            # flipped [::-1,::-1] to match train_utils_sim.obs_to_img convention.
            if 'libero' in variant.env:
                chunk_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            elif 'aloha' in variant.env:
                chunk_frames.append(np.ascontiguousarray(obs["pixels"]["top"]))
            chunk_env_steps.append(t)

            # For scattered sampling we need a per-env-step lifted-noise stream.
            # Stash this chunk's (query_freq, noise_dim) latents — collect_traj
            # only constructs the stream array at the end.
            if record_scattered:
                ns = np.asarray(actions_noise, dtype=np.float32)
                if ns.ndim != 2 or ns.shape[0] != query_frequency:
                    raise RuntimeError(
                        f"scattered_mode requires lifted noise (sac_action_chunk_size=={query_frequency}); "
                        f"got actions_noise shape {ns.shape}"
                    )
                scattered_noise_chunks.append(ns)

        action_t = actions[t % query_frequency]
        if record_scattered:
            scattered_obs_stream.append({k: np.asarray(v) for k, v in obs_dict.items()})
            scattered_executed_stream.append(np.asarray(action_t, dtype=np.float32))
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated

        rewards.append(reward)
        image_list.append(curr_image)
        if done and libero_success_step is None:
            libero_success_step = t
        # Do NOT break early — robometer needs the full-horizon video and the
        # reward fn may rely on per-step libero rewards after success too.

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
        'state': qpos[np.newaxis, ..., np.newaxis],
    }
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards_per_step = np.array(rewards)
    episode_return = float(np.sum(rewards_per_step[rewards_per_step != None]))
    is_success = libero_success_step is not None or (reward == env_max_reward)
    # Use tqdm.write so the message doesn't tear the active progress bar
    # (which leaves frozen mid-redraws like "388/400" stranded above it).
    tqdm.write(
        f'Rollout Done: {episode_return=}, Success: {is_success}, '
        f'libero_success_step={libero_success_step}'
    )

    # NOTE: rewards/masks for the SAC buffer are computed downstream by the
    # reward_fn from robometer scores. collect_traj intentionally leaves those
    # out so train_sim_robometer's async queue is the single source of truth.
    out = {
        'observations': obs_list,
        'actions': action_list,                  # noise latents (DSRL-SAC primary, DSRL-NA's noise critic input)
        'diffused_actions': diffused_action_list,  # pi0 env-actions (legacy v0 field; NA path also stores executed_actions below)
        'executed_actions': diffused_action_list,  # alias matching reference DSRL-NA helper expectations
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1,
        'libero_rewards_per_step': rewards_per_step.astype(np.float32),
        'libero_success_step': libero_success_step,
        'chunk_frames': np.stack(chunk_frames, axis=0) if chunk_frames else np.zeros((0, 0, 0, 3), dtype=np.uint8),
        'chunk_env_steps': np.asarray(chunk_env_steps, dtype=np.int64),
    }
    if is_na:
        out['original_observations'] = original_obs_list
        if save_kv_cache and k_cache_outs:
            out['original_k_cache'] = k_cache_outs
            out['original_v_cache'] = v_cache_outs
    if record_scattered:
        T = len(scattered_obs_stream)
        if scattered_noise_chunks:
            noise_stream = np.concatenate(scattered_noise_chunks, axis=0)[:T]
        else:
            noise_stream = np.zeros((T, 0), dtype=np.float32)
        executed_stream = (
            np.stack(scattered_executed_stream, axis=0)
            if scattered_executed_stream
            else np.zeros((T, 0), dtype=np.float32)
        )
        out['scattered_obs_stream'] = scattered_obs_stream
        out['scattered_noise_stream'] = noise_stream
        out['scattered_executed_stream'] = executed_stream
    return out

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        rewards = []
        

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                qpos = obs_to_qpos(obs, variant)
                if variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    }

                rng, key = jax.random.split(rng)
                assert agent_dp is not None
                
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                
                
                if i == 0:
                    # for initial evaluation, we sample from standard gaussian noise to evaluate the base policy's performance
                    noise = jax.random.normal(rng, (1, 50, 32))
                else:
                    # NOTE(future change): eval uses the stochastic SAC actor
                    # (`sample_actions` == dist.sample). For a cleaner eval
                    # metric we'd swap this to the deterministic mode:
                    #     actions_noise = agent.eval_actions(obs_dict)
                    # which calls eval_actions_jit -> dist.mode() (see
                    # jaxrl2/agents/agent.py:24 and jaxrl2/agents/common.py:74).
                    # That removes actor-policy sample variance from the
                    # reported success rate so curves reflect the policy's
                    # best action, not IID latent draws. Leaving stochastic
                    # for now to match the original train_utils_sim behavior.
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
                    
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        tqdm.write(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
