#!/bin/bash
set -euo pipefail

# Single-episode pi0 rollout on LIBERO, scored against the Robometer eval server.
# Mirrors run_libero.sh's env setup (EGL rendering, MUJOCO, OpenPI paths).

proj_name=DSRL_pi0_Robometer_Test
# Robometer server is assumed to hold GPU 0, so default pi0 to GPU 1. Override
# with DEVICE_ID=N if your setup differs.
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
TASK_ID=${TASK_ID:-57}
SEED=${SEED:-0}
QUERY_FREQ=${QUERY_FREQ:-20}
MAX_TIMESTEPS=${MAX_TIMESTEPS:-400}
OUTPUT_DIR=${OUTPUT_DIR:-$EXP/rollout}

uv run examples/rollout_robometer.py \
    --env libero \
    --task_id "$TASK_ID" \
    --seed "$SEED" \
    --query_freq "$QUERY_FREQ" \
    --max_timesteps "$MAX_TIMESTEPS" \
    --camera_resolution 256 \
    --robometer_url "$ROBOMETER_URL" \
    --output_dir "$OUTPUT_DIR"
