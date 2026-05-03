#!/bin/bash
set -euo pipefail

# DSRL pi0 LIBERO + robometer dense-reward training, **DSRL-NA two-critic variant**.
#
# Sibling of run_libero_robometer.sh — same env setup, same robometer plumbing,
# only difference is `--algorithm pixel_dsrl_na` plus the noise-critic distillation
# knob `--noise_critic_grad_steps`. Reward kinds, success threshold, and queue
# depth all carry over from the env vars below; tweak as you would for the
# DSRL-SAC script.
#
# See docs/dsrl_na_integration_plan.md for the algorithm spec and the rationale
# for the v0 stale-target source.

proj_name=DSRL_pi0_Libero_Robometer_NA
device_id=${DEVICE_ID:-1}
export WANDB_ENTITY=peterchenyipu

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${PYTHONPATH:-}:."

ROBOMETER_URL=${ROBOMETER_URL:-http://localhost:8000}
ROBOMETER_REWARD_KIND=${ROBOMETER_REWARD_KIND:-robo_success_plus_robo_progress}
ROBOMETER_SUCCESS_THRESHOLD=${ROBOMETER_SUCCESS_THRESHOLD:-0.5}
ROBOMETER_QUEUE_MAX_DEPTH=${ROBOMETER_QUEUE_MAX_DEPTH:-4}

# DSRL-NA knobs aligned to the arayabrain reference's run_aloha_sim_na.sh.
# Ref values commented inline. Override via env vars to ablate.
SAVE_KV_CACHE=${SAVE_KV_CACHE:-1}                     # ref: 1
GRL_NOISE_SAMPLE=${GRL_NOISE_SAMPLE:-1}               # ref: 1
NOISE_SCALE_INSIDE=${NOISE_SCALE_INSIDE:-0}           # ref: 0 (multiply externally)
DSRL_NA_BACKUP_ENTROPY=${DSRL_NA_BACKUP_ENTROPY:-0}   # ref: 0
ACTION_CRITIC_STEPS=${ACTION_CRITIC_STEPS:-15}        # ref: 15
NOISE_CRITIC_STEPS=${NOISE_CRITIC_STEPS:-5}           # ref: 5
NOISE_ACTOR_STEPS=${NOISE_ACTOR_STEPS:-5}             # ref: 5
FLOW_INTEGRATION_STEPS=${FLOW_INTEGRATION_STEPS:-10}  # ref: 10
# Host-RAM-bounded for our 60 GB box (robometer ≈14 GB RSS already): each slot
# holds ~18 MB of K/V cache (CPU-pinned), so 1000 slots ≈ 18 GB. Reference's
# 150_000 would need ~2.7 TB — impossible on this hardware. See
# docs/dsrl_na_cli_args.md for the breakdown; override via env var if you have
# more RAM.
ONLINE_BUFFER_SIZE=${ONLINE_BUFFER_SIZE:-1000}        # ref: 150_000 — RAM-bounded here
NUM_INITIAL_TRAJ_COLLECT=${NUM_INITIAL_TRAJ_COLLECT:-25}   # ref: 125 — half of 50-traj buffer capacity
TARGET_ENTROPY=${TARGET_ENTROPY:-0.0}                 # ref: 0.0
SAC_ACTION_CHUNK_SIZE=${SAC_ACTION_CHUNK_SIZE:-1}     # 1 = unlifted (ref); set to query_freq for lift
PUT_KV_CACHE_ON_CPU=${PUT_KV_CACHE_ON_CPU:-1}         # ref: 1 (host-RAM K/V to fit 150k buffer)

uv pip install mujoco==3.3.1

# DSRL-NA learner: SAC actor outputs (sac_action_chunk_size, 32) noise (default
# 1×32, matching the arayabrain reference). The action critic sees (query_freq,
# 7) diffused env-actions; the noise critic is distilled from the action critic.
# Per outer SAC step, pi0 IS run inside _na_update_step (action_critic_steps
# times for the Bellman target + noise_critic_steps times for distillation
# pairs). With save_kv_cache=1 the prompt-encoder K/V is reused, so each pi0
# call is just the diffusion stack.
#
# Reference values (run_aloha_sim_na.sh): batch=128, query_freq=50, num_qs=10,
# action_magnitude=0.75, target_entropy=0.0, action_critic_steps=15,
# noise_critic_steps=5, noise_actor_steps=5, flow_integration_steps=10,
# online_buffer_size=150_000, num_initial_traj_collect=125,
# grl_noise_sample=1, backup_entropy=0, noise_scale_inside=0.
#
# Local override: online_buffer_size=1000 (≈18 GB host RAM) +
# num_initial_traj_collect=25 (~500 slots = half-buffer warmup) so we don't
# OOM on the 60 GB box where robometer already holds ~14 GB RSS.
# Step-counting divergence from the reference: ours' pbar ticks per OUTER SAC
# step (= action_critic_steps + noise_critic_steps + noise_actor_steps = 25
# inner gradient updates), while the reference ticks per inner step. To match
# the reference's run_aloha_sim_na.sh training budget (1M inner updates,
# eval every 5000 inner, log every 100 inner, ckpt every 10000 inner), divide
# each of the *_steps / *_interval flags by 25. start_online_updates is a
# buffer-transition threshold (NOT a gradient-step count) — leave it alone.
uv run examples/launch_train_sim_robometer.py \
--algorithm pixel_dsrl_na \
--env libero \
--prefix dsrl_pi0_libero_na \
--wandb_project ${proj_name} \
--batch_size 128 \
--discount 0.999 \
--seed 0 \
--max_steps 40000 \
--eval_interval 200 \
--log_interval 4 \
--eval_episodes 10 \
--checkpoint_interval 400 \
--multi_grad_step 1 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 0.75 \
--query_freq 20 \
--hidden_dims 128 \
--target_entropy "$TARGET_ENTROPY" \
--num_qs 10 \
--sac_action_chunk_size "$SAC_ACTION_CHUNK_SIZE" \
--save_kv_cache "$SAVE_KV_CACHE" \
--put_kv_cache_on_cpu "$PUT_KV_CACHE_ON_CPU" \
--grl_noise_sample "$GRL_NOISE_SAMPLE" \
--noise_scale_inside "$NOISE_SCALE_INSIDE" \
--dsrl_na_backup_entropy "$DSRL_NA_BACKUP_ENTROPY" \
--action_critic_steps "$ACTION_CRITIC_STEPS" \
--noise_critic_steps "$NOISE_CRITIC_STEPS" \
--noise_actor_steps "$NOISE_ACTOR_STEPS" \
--flow_integration_steps "$FLOW_INTEGRATION_STEPS" \
--online_buffer_size "$ONLINE_BUFFER_SIZE" \
--num_initial_traj_collect "$NUM_INITIAL_TRAJ_COLLECT" \
--robometer_url "$ROBOMETER_URL" \
--robometer_reward_kind "$ROBOMETER_REWARD_KIND" \
--robometer_success_threshold "$ROBOMETER_SUCCESS_THRESHOLD" \
--robometer_queue_max_depth "$ROBOMETER_QUEUE_MAX_DEPTH"
