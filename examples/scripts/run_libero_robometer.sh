#!/bin/bash
set -euo pipefail

# DSRL pi0 LIBERO + robometer dense-reward training.
# Assumes a Robometer eval server is running at ${ROBOMETER_URL} (default
# http://localhost:8000). The robometer server is expected to hold GPU 0, so
# pi0+SAC default to GPU 1 — override with DEVICE_ID=N.

proj_name=DSRL_pi0_Libero_Robometer
device_id=${DEVICE_ID:-1}

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

# ----------------------------------------------------------------------------
# Reward function (ROBOMETER_REWARD_KIND). Pick ONE:
#
#   libero_success_plus_robo_progress  [DEFAULT]
#       base = -1 per chunk, 0 at terminal if libero reports success; then
#       + robometer.progress[t] added on top. Falls back to libero-sparse
#       on server failure.
#
#   robo_success_plus_robo_progress
#       Same shape as above, but "success" comes from the robometer success
#       head: any(success >= ROBOMETER_SUCCESS_THRESHOLD) flips the terminal.
#       Libero's is_success is ignored (useful when libero reward is noisy
#       or unavailable). Falls back to libero-sparse if the server returns
#       no success head.
#
#   robo_success_sparse
#       Same shape as libero_sparse (-1 per chunk, 0 at terminal on success),
#       but success comes from the robometer success head + threshold. No
#       progress shaping. Falls back to libero-detected success if the
#       server returned no success head.
#
#   progress_delta
#       r[t] = progress[t+1] - progress[t]; last entry = 0. Dense shaping
#       reward that averages to progress[-1] - progress[0] over an episode.
#
#   progress
#       r[t] = progress[t] directly (level, not delta).
#
#   success
#       r[t] = robometer.success[t]. Requires a success head on the server.
#
#   libero_sparse
#       Original DSRL pi0 reward: -1 per chunk, 0 on terminal success.
#       No robometer dependency; useful as a training-from-scratch baseline.
# ----------------------------------------------------------------------------
ROBOMETER_REWARD_KIND=${ROBOMETER_REWARD_KIND:-robo_success_plus_robo_progress}
ROBOMETER_SUCCESS_THRESHOLD=${ROBOMETER_SUCCESS_THRESHOLD:-0.5}
ROBOMETER_QUEUE_MAX_DEPTH=${ROBOMETER_QUEUE_MAX_DEPTH:-4}

uv pip install mujoco==3.3.1

# SAC action shape is (query_freq, 32) — one 32-d noise latent PER env-step
# in the chunk (was (1, 32) broadcast before the lift). Actor head outputs
# query_freq * 32 = 640 values; bumping --query_freq grows the actor head
# proportionally. Paper-recommended config is much wider — see
# docs/sac_action_lift_plan.md (e.g. --hidden_dims 2048 2048 2048 and
# --action_magnitude 1.5).
uv run examples/launch_train_sim_robometer.py \
--algorithm pixel_sac \
--env libero \
--prefix dsrl_pi0_libero_robometer \
--wandb_project ${proj_name} \
--batch_size 256 \
--discount 0.999 \
--seed 0 \
--max_steps 500000  \
--eval_interval 10000 \
--log_interval 500 \
--eval_episodes 10 \
--multi_grad_step 20 \
--start_online_updates 500 \
--resize_image 64 \
--action_magnitude 1.0 \
--query_freq 20 \
--hidden_dims 128 \
--robometer_url "$ROBOMETER_URL" \
--robometer_reward_kind "$ROBOMETER_REWARD_KIND" \
--robometer_success_threshold "$ROBOMETER_SUCCESS_THRESHOLD" \
--robometer_queue_max_depth "$ROBOMETER_QUEUE_MAX_DEPTH"
