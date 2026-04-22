from tqdm import tqdm
import numpy as np
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
    """reward[b, t] = progress[b, t+1] - progress[b, t]; last column = 0."""
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
    """reward[b, t] = progress[b, t] directly (level reward)."""
    B, T = _validate_batch(scores, metas)
    rewards = _require_progress_len(scores, T)
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        if bool(m.get("libero_is_success", False)):
            masks[b, -1] = 0.0
    return rewards, masks


def _success_batched(scores: List[ScoreResult], metas: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """reward[b, t] = success_prob[b, t]. Raises if any episode has no success head."""
    B, T = _validate_batch(scores, metas)
    rewards = _require_success_len(scores, T)
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        if bool(m.get("libero_is_success", False)):
            masks[b, -1] = 0.0
    return rewards, masks


def _libero_success_plus_robo_progress_batched(
    scores: List[ScoreResult], metas: List[Dict]
) -> Tuple[np.ndarray, np.ndarray]:
    """Libero sparse base + robometer progress. Success oracle: libero.

    reward[b, t] = -1 + progress[b, t]; terminal -> 0 + progress[b, -1] on libero success.
    """
    B, T = _validate_batch(scores, metas)
    progress = _require_progress_len(scores, T)
    rewards = np.full((B, T), -1.0, dtype=np.float32) + progress
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        if bool(m.get("libero_is_success", False)):
            rewards[b, -1] = 0.0 + progress[b, -1]
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
    """Sparse base + progress, success oracle from robometer success head + threshold.

    Requires progress AND success on every ScoreResult. Libero is_success is ignored.
    """
    B, T = _validate_batch(scores, metas)
    progress = _require_progress_len(scores, T)
    success = _require_success_len(scores, T)
    rewards = np.full((B, T), -1.0, dtype=np.float32) + progress
    masks = np.ones((B, T), dtype=np.float32)
    for b, m in enumerate(metas):
        threshold = float(m.get("robometer_success_threshold", 0.5))
        if bool(np.any(success[b] >= threshold)):
            rewards[b, -1] = 0.0 + progress[b, -1]
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
    if shard_fn is not None:
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

        def _insert_batched(resolved_pairs):
            """Apply reward_fn to a batch of resolved (traj, score) pairs and insert."""
            nonlocal total_env_steps
            if not resolved_pairs:
                return
            scored = _apply_reward_fn(
                reward_fn, resolved_pairs, variant.robometer_success_threshold
            )
            for traj_done, rewards, masks, score in scored:
                _insert_scored_traj(variant, traj_done, rewards, masks, score,
                                    online_replay_buffer, wandb_logger, i)
                total_env_steps += traj_done['env_steps']

        with tqdm(total=variant.max_steps, initial=0) as pbar:
            while i <= variant.max_steps:
                # 1) Collect a new rollout (blocking physics sim).
                traj = collect_traj(variant, agent, env, i, agent_dp)
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
                if len(online_replay_buffer) <= variant.start_online_updates and len(pending) > 0:
                    _insert_batched(pending.drain_all())

                traj_id = online_replay_buffer._traj_counter
                print('online buffer timesteps length:', len(online_replay_buffer))
                print('online buffer num traj:', traj_id)
                print('total env steps:', total_env_steps)
                print('robometer queue depth:', len(pending), 'stats:', pending.stats())

                if variant.get("num_online_gradsteps_batch", -1) > 0:
                    num_gradsteps = variant.num_online_gradsteps_batch
                else:
                    num_gradsteps = len(traj["actions"]) * variant.multi_grad_step

                if len(online_replay_buffer) > variant.start_online_updates:
                    for _ in range(num_gradsteps):
                        if i == 0:
                            print('performing evaluation for initial checkpoint')
                            if perform_control_evals:
                                perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                            if hasattr(agent, 'perform_eval'):
                                agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                        batch = next(replay_buffer_iterator)
                        update_info = agent.update(batch)

                        pbar.update()
                        i += 1

                        if i % variant.log_interval == 0:
                            update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                            for k, v in update_info.items():
                                if v.ndim == 0:
                                    wandb_logger.log({f'training/{k}': v}, step=i)
                                elif v.ndim <= 2:
                                    wandb_logger.log_histogram(f'training/{k}', v, i)
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
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.asarray(rewards, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.float32)
    assert len(rewards) == episode_len, f"rewards len {len(rewards)} != {episode_len}"
    assert len(masks) == episode_len, f"masks len {len(masks)} != {episode_len}"

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
    action_list = []
    obs_list = []
    chunk_frames: List[np.ndarray] = []   # 256x256 agentview frames at chunk boundaries
    chunk_env_steps: List[int] = []

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

            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)
            # 256x256 chunk-boundary frame for robometer. Libero's agentview is
            # flipped [::-1,::-1] to match train_utils_sim.obs_to_img convention.
            if 'libero' in variant.env:
                chunk_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            elif 'aloha' in variant.env:
                chunk_frames.append(np.ascontiguousarray(obs["pixels"]["top"]))
            chunk_env_steps.append(t)

        action_t = actions[t % query_frequency]
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
    print(f'Rollout Done: {episode_return=}, Success: {is_success}, '
          f'libero_success_step={libero_success_step}')

    # NOTE: rewards/masks for the SAC buffer are computed downstream by the
    # reward_fn from robometer scores. collect_traj intentionally leaves those
    # out so train_sim_robometer's async queue is the single source of truth.
    return {
        'observations': obs_list,
        'actions': action_list,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1,
        'libero_rewards_per_step': rewards_per_step.astype(np.float32),
        'libero_success_step': libero_success_step,
        'chunk_frames': np.stack(chunk_frames, axis=0) if chunk_frames else np.zeros((0, 0, 0, 3), dtype=np.uint8),
        'chunk_env_steps': np.asarray(chunk_env_steps, dtype=np.int64),
    }

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
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
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
  
