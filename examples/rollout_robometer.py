#! /usr/bin/env python
"""Roll one libero episode under pi0 and score the video with a Robometer server.

Intentionally minimal: no SAC learner, no replay buffer, no wandb. The goal is to
collect, for a single rollout:

  * per-env-step libero reward (length T)
  * per-env-step 256x256 agentview frame (length T)

then send the video to the Robometer eval server and plot libero reward vs.
Robometer per-frame progress (and success, if the server exposes it).

Run via ``examples/scripts/test_robometer.sh`` (sets MUJOCO_GL=egl etc).
"""
from __future__ import annotations

import os

# Match train_sim.py: opt into Triton GEMM before jax imports.
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

import argparse
import pathlib
import types
from dataclasses import dataclass
from typing import Any, Dict, List

import imageio.v2 as imageio
import jax
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from tqdm import tqdm

import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as openpi_config

from examples.train_utils_sim import obs_to_pi_zero_input

from robometer_client import RobometerClient, VideoSample

home_dir = os.environ["HOME"]
compilation_cache.set_cache_dir(os.path.join(home_dir, "jax_compilation_cache"))


@dataclass
class RolloutConfig:
    env: str = "libero"
    task_id: int = 57
    seed: int = 0
    max_timesteps: int = 400
    query_freq: int = 20
    action_chunk_size: int = 50  # pi0 internal chunk length used as noise shape
    sac_noise_chunk: int = 1     # matches train_sim i==0: SAC emits a (1, 32) latent
    noise_dim: int = 32
    camera_resolution: int = 256
    robometer_url: str = "http://localhost:8000"
    output_dir: str = "./logs/robometer_test"
    # Robometer is scored only at SAC chunk boundaries (one frame per query_freq env
    # steps) — that's the cadence the SAC buffer actually sees, so scoring any more
    # is wasted. rm_frame_size is the spatial size robometer expects.
    use_frame_steps: bool = True
    rm_frame_size: int = 224
    robometer_success_threshold: float = 1.0


def _get_libero_env(task, resolution: int, seed: int):
    """Initialize LIBERO env for the given task (mirrors train_sim._get_libero_env)."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _agentview_frame(obs: Dict[str, Any]) -> np.ndarray:
    """Return the raw 256x256 RGB agentview frame, flipped to match train_utils_sim."""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def collect_rollout(
    cfg: RolloutConfig,
    env,
    agent_dp,
    task_description: str,
) -> Dict[str, Any]:
    """Roll a single episode for the full horizon — never break early on success.

    Mirrors ``train_utils_sim.collect_traj`` at i==0 (pre-SAC): every ``query_freq``
    env steps we sample a single ``(sac_noise_chunk, noise_dim)`` Gaussian latent
    and broadcast it across pi0's 50-step diffusion to produce the next action
    chunk. This is what ``train_sim`` calls the "base pi0" rollout.

    Returns both the dense per-env-step data (reward, frame) and the chunk-boundary
    256×256 frames that align 1-to-1 with SAC obs buffer entries.
    """
    rng = jax.random.PRNGKey(cfg.seed)
    obs = env.reset()

    frames: List[np.ndarray] = []
    rewards: List[float] = []
    # Chunk-boundary data, matched 1-to-1 with what the SAC buffer would store.
    chunk_frames: List[np.ndarray] = []
    chunk_env_steps: List[int] = []
    success_step: int | None = None

    actions = None
    for t in tqdm(range(cfg.max_timesteps), desc="rollout"):
        # Record raw-camera frame BEFORE stepping so len(frames) == len(rewards).
        frames.append(_agentview_frame(obs))

        if t % cfg.query_freq == 0:
            rng, key = jax.random.split(rng)
            variant_stub = types.SimpleNamespace(env=cfg.env, task_description=task_description)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant_stub)

            # train_sim i==0 noise: (1, sac_noise_chunk, noise_dim), tail-padded to
            # pi0's full 50-step diffusion horizon by repeating the last row.
            base_noise = jax.random.normal(
                key, (1, cfg.sac_noise_chunk, cfg.noise_dim)
            )
            pad = jax.numpy.repeat(
                base_noise[:, -1:, :], cfg.action_chunk_size - cfg.sac_noise_chunk, axis=1
            )
            noise = jax.numpy.concatenate([base_noise, pad], axis=1)
            actions = agent_dp.infer(obs_pi_zero, noise=np.asarray(noise))["actions"]

            # Save the chunk-boundary frame — this is the one-to-one analogue of
            # what ``add_online_data_to_buffer`` inserts into the SAC buffer.
            chunk_frames.append(_agentview_frame(obs))
            chunk_env_steps.append(t)

        action_t = actions[t % cfg.query_freq]
        obs, reward, done, _ = env.step(action_t)
        rewards.append(float(reward))

        # Record success onset but DON'T break — we want the full horizon so
        # robometer can annotate past-success chunks too.
        if done and success_step is None:
            success_step = t

    return {
        "frames": np.stack(frames, axis=0),                      # (T, 256, 256, 3)
        "rewards": np.asarray(rewards, dtype=np.float32),        # (T,)
        "chunk_frames": np.stack(chunk_frames, axis=0),          # (T/query_freq, 256, 256, 3)
        "chunk_env_steps": np.asarray(chunk_env_steps, dtype=np.int64),
        "success_step": success_step,
        "task": task_description,
    }


def _resize_video(frames: np.ndarray, frame_size: int) -> np.ndarray:
    """Resize every frame in (T, H, W, C) to (T, frame_size, frame_size, C) with BILINEAR."""
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


def score_with_robometer(
    cfg: RolloutConfig,
    chunk_frames: np.ndarray,
    task: str,
) -> Dict[str, np.ndarray | None]:
    """Send the ``T/query_freq`` chunk-boundary frames to the Robometer eval server.

    With ``use_frame_steps=True`` the server returns a length-N list of scalars where
    entry k is the progress inferred from the prefix frames[0..k]. Frames are
    resized to ``rm_frame_size`` (the server's expected spatial input).
    """
    video = _resize_video(chunk_frames, cfg.rm_frame_size)
    with RobometerClient(cfg.robometer_url) as client:
        print("robometer health:", client.health())
        sample = VideoSample(frames=video, task=task, id="rollout_0")
        [result] = client.score([sample], use_frame_steps=cfg.use_frame_steps)
    progress = np.asarray(result.progress, dtype=np.float32)
    success = None if result.success is None else np.asarray(result.success, dtype=np.float32)
    return {"progress": progress, "success": success}


def _first_index_ge(arr: np.ndarray, threshold: float) -> int | None:
    """First index where arr >= threshold, or None."""
    hits = np.where(arr >= threshold)[0]
    return int(hits[0]) if hits.size else None


def _image_strip(frames: np.ndarray) -> np.ndarray:
    """Concatenate every frame in (T, H, W, 3) horizontally into one strip."""
    return np.concatenate(list(frames), axis=1)


def plot_signals(
    cfg: RolloutConfig,
    rewards: np.ndarray,
    chunk_env_steps: np.ndarray,
    chunk_frames: np.ndarray,
    progress: np.ndarray,
    success: np.ndarray | None,
    libero_success_step: int | None,
    output_dir: str,
    task: str = "",
) -> None:
    """4-row figure mirroring jaxrl2's ``make_visual`` (image strip + value traces).

    Rows:
      0. horizontal strip of sampled chunk-boundary frames
      1. robometer per-chunk progress
      2. robometer per-chunk success probability (if available)
      3. libero per-env-step reward

    Two vertical markers are drawn on all non-image rows:
      * green dashed = libero env success step
      * red  dashed = first chunk where robometer success >= threshold
    """
    os.makedirs(output_dir, exist_ok=True)

    # Per-chunk x axis (env-step indices where the chunk started).
    x_chunk = np.asarray(chunk_env_steps[: len(progress)], dtype=np.int64)
    x_env = np.arange(len(rewards))
    robometer_success_step = None
    if success is not None:
        k = _first_index_ge(success, cfg.robometer_success_threshold)
        if k is not None:
            robometer_success_step = int(x_chunk[k])

    def _mark(ax):
        if libero_success_step is not None:
            ax.axvline(libero_success_step, color="tab:green", linestyle="--",
                       label=f"libero success @ t={libero_success_step}")
        if robometer_success_step is not None:
            ax.axvline(robometer_success_step, color="tab:red", linestyle="--",
                       label=f"robometer success >={cfg.robometer_success_threshold:.2f} @ t={robometer_success_step}")
        ax.legend(loc="best", fontsize=8)

    # Layout strategy:
    #   * each chunk-boundary frame is square (1:1 pixel aspect);
    #   * all signal plots share the same env-step x-axis as the image strip, so a
    #     vertical line through the figure hits "image N ↔ env step M" exactly.
    # Concretely: pick a per-frame display size (inches) and let fig_w track the
    # number of chunks. ``set_box_aspect(1/num_chunks)`` forces axs[0]'s bounding
    # box to num_chunks:1, keeping frames square regardless of fig_h.
    num_chunks = len(chunk_frames)
    per_frame_in = 1.2
    signal_row_in = 1.6
    margin_in = 2.0
    fig_w = max(12.0, per_frame_in * num_chunks + margin_in)
    fig_h = per_frame_in + signal_row_in * 3 + margin_in
    fig, axs = plt.subplots(
        4, 1, figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [per_frame_in, signal_row_in, signal_row_in, signal_row_in]},
        sharex=True,
    )
    if task:
        fig.suptitle(f'task #{cfg.task_id}: "{task}"', fontsize=14, y=0.995)

    strip = _image_strip(chunk_frames)
    strip_left = float(chunk_env_steps[0])
    strip_right = float(chunk_env_steps[-1] + cfg.query_freq)
    axs[0].imshow(strip, extent=(strip_left, strip_right, 0, 1), aspect="auto")
    axs[0].set_box_aspect(1.0 / num_chunks)
    axs[0].set_yticks([])
    # Tick at the left edge of each chunk so user can read off exact env step.
    chunk_edges = np.concatenate([chunk_env_steps, [chunk_env_steps[-1] + cfg.query_freq]])
    for e in chunk_edges:
        axs[0].axvline(e, color="white", linewidth=0.5, alpha=0.35)
    axs[0].set_title(
        f"Chunk-boundary frames ({len(chunk_frames)} chunks, query_freq={cfg.query_freq})"
    )

    axs[1].plot(x_chunk, progress, linestyle="--", marker="o", color="tab:orange", label="robometer progress")
    axs[1].set_ylabel("progress")
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].grid(True, alpha=0.3)
    _mark(axs[1])

    if success is not None:
        axs[2].plot(x_chunk, success, linestyle="--", marker="o", color="tab:red", label="robometer success prob.")
    else:
        axs[2].text(0.5, 0.5, "server did not return success probs",
                    ha="center", va="center", transform=axs[2].transAxes)
    axs[2].set_ylabel("success prob.")
    axs[2].set_ylim(-0.05, 1.05)
    axs[2].grid(True, alpha=0.3)
    _mark(axs[2])

    axs[3].plot(x_env, rewards, color="tab:blue", label="libero reward (per env step)")
    axs[3].set_ylabel("libero reward")
    axs[3].set_xlabel("env step")
    axs[3].set_ylim(-0.05, 1.05)
    axs[3].grid(True, alpha=0.3)
    _mark(axs[3])

    # Dense x-ticks at every chunk boundary so you can read off "image N ↔ step M".
    axs[-1].set_xticks(chunk_edges)
    axs[-1].set_xticklabels([str(int(e)) for e in chunk_edges], rotation=45, ha="right", fontsize=8)
    for ax in axs[1:]:
        ax.set_xlim(strip_left, strip_right)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(output_dir, "robometer_vs_libero.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_video(frames: np.ndarray, path: str, fps: int = 30) -> None:
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


def main(cfg: RolloutConfig) -> None:
    tf.config.set_visible_devices([], "GPU")

    if cfg.env != "libero":
        raise NotImplementedError("this test script currently only wires up libero")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_90"]()
    task = task_suite.get_task(cfg.task_id)
    env, task_description = _get_libero_env(task, cfg.camera_resolution, cfg.seed)
    print(f"task {cfg.task_id}: {task_description}")

    config = openpi_config.get_config("pi0_libero")
    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    print("loaded pi0 policy from", checkpoint_dir)

    rollout = collect_rollout(cfg, env, agent_dp, task_description)
    frames: np.ndarray = rollout["frames"]
    rewards: np.ndarray = rollout["rewards"]
    chunk_frames: np.ndarray = rollout["chunk_frames"]
    chunk_env_steps: np.ndarray = rollout["chunk_env_steps"]
    libero_success_step = rollout["success_step"]
    print(
        f"rollout done: T={len(frames)} env steps, {len(chunk_frames)} chunk-boundary frames, "
        f"reward_sum={rewards.sum():.2f}, libero_success_step={libero_success_step}"
    )

    # Persist raw rollout immediately — robometer scoring is the cheap part to retry.
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_video(frames, os.path.join(cfg.output_dir, "rollout.mp4"))
    np.savez(
        os.path.join(cfg.output_dir, "rollout_raw.npz"),
        frames=frames,
        rewards=rewards,
        chunk_frames=chunk_frames,
        chunk_env_steps=chunk_env_steps,
        success_step=np.array([-1 if libero_success_step is None else libero_success_step]),
        task=np.array([task_description]),
    )

    scores = score_with_robometer(cfg, chunk_frames, task_description)
    progress = scores["progress"]
    success = scores["success"]
    print(
        f"robometer: N={len(progress)} chunk scores, "
        f"final_progress={float(progress[-1]):.3f}"
        + (f", final_success={float(success[-1]):.3f}" if success is not None else "")
    )

    np.savez(
        os.path.join(cfg.output_dir, "rollout.npz"),
        rewards=rewards,
        chunk_env_steps=chunk_env_steps,
        progress=progress,
        success=success if success is not None else np.array([]),
        libero_success_step=np.array([-1 if libero_success_step is None else libero_success_step]),
        task=np.array([task_description]),
    )
    plot_signals(
        cfg,
        rewards=rewards,
        chunk_env_steps=chunk_env_steps,
        chunk_frames=chunk_frames,
        progress=progress,
        success=success,
        libero_success_step=libero_success_step,
        output_dir=cfg.output_dir,
        task=task_description,
    )
    print(f"wrote outputs to {cfg.output_dir}")


def _parse_args() -> RolloutConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="libero")
    p.add_argument("--task_id", type=int, default=57)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_timesteps", type=int, default=400)
    p.add_argument("--query_freq", type=int, default=20)
    p.add_argument("--camera_resolution", type=int, default=256)
    p.add_argument("--robometer_url", default="http://localhost:8000")
    p.add_argument("--output_dir", default="./logs/robometer_test")
    p.add_argument("--use_frame_steps", type=int, default=1,
                   help="1 = causal per-chunk progress (one server pass per prefix); "
                        "0 = single progress sequence for the whole video")
    p.add_argument("--rm_frame_size", type=int, default=224,
                   help="Spatial resolution for robometer frames; matches the example's default")
    p.add_argument("--robometer_success_threshold", type=float, default=1.0,
                   help="Threshold on robometer success prob for drawing the red success marker")
    args = p.parse_args()
    return RolloutConfig(
        env=args.env,
        task_id=args.task_id,
        seed=args.seed,
        max_timesteps=args.max_timesteps,
        query_freq=args.query_freq,
        camera_resolution=args.camera_resolution,
        robometer_url=args.robometer_url,
        output_dir=args.output_dir,
        use_frame_steps=bool(args.use_frame_steps),
        rm_frame_size=args.rm_frame_size,
        robometer_success_threshold=args.robometer_success_threshold,
    )


if __name__ == "__main__":
    main(_parse_args())
