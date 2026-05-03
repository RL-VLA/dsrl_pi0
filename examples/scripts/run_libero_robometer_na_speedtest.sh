#!/bin/bash
set -euo pipefail

# Speed-comparison smoke for DSRL-NA: matches the arayabrain reference impl's
# /data3/dsrl_pi0_na/examples/scripts/run_aloha_sim_na.sh hyperparameter set
# (batch=16, query_freq=50, num_qs=10, hidden_dims=128, save_kv_cache=1, etc.)
# so wall-clock per SAC update step can be compared apples-to-apples.
#
# Differences from the reference that ARE NOT hyperparameters:
#   - env is libero (not aloha_cube): physics cost differs but is rollout-only.
#   - dense reward via robometer (reference uses dense env reward).
#
# `--sac_action_chunk_size 1` matches the reference's `(1, 32)` SAC noise shape
# (no action-space lift), so the SAC actor heads have identical output dim.
# Override on the command line to re-run with the lift enabled.

proj_name=DSRL_pi0_Libero_Robometer_NA_speedtest
device_id=${DEVICE_ID:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}

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

# Hyperparams aligned to the reference smoke (run_ref_na_smoke.sh):
#   batch_size=16, query_freq=50, multi_grad_step=1, hidden_dims=128,
#   action_magnitude=0.75, save_kv_cache=1, noise_scale_inside=0,
#   grl_noise_sample=0, dsrl_na_backup_entropy=0, max_steps=30, eval_episodes=1.
uv run examples/launch_train_sim_robometer.py \
    --algorithm pixel_dsrl_na \
    --env libero \
    --prefix na_speedtest \
    --wandb_project ${proj_name} \
    --batch_size 16 --discount 0.999 --seed 0 \
    --max_steps 30 --eval_interval 999999 --log_interval 5 --eval_episodes 1 \
    --multi_grad_step 1 --start_online_updates 1 \
    --resize_image 64 --action_magnitude 0.75 --query_freq 50 --hidden_dims 128 \
    --sac_action_chunk_size 1 \
    --noise_critic_grad_steps 1 \
    --save_kv_cache 1 \
    --noise_scale_inside 0 \
    --grl_noise_sample 0 \
    --dsrl_na_backup_entropy 0 \
    --robometer_url "$ROBOMETER_URL" \
    --robometer_reward_kind libero_success_plus_robo_progress \
    --robometer_success_threshold 0.5 \
    --robometer_queue_max_depth 4
