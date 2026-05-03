# Setup Guide — DSRL pi0

End-to-end environment setup for the DSRL pi0 (libero / aloha sim) project,
including the openpi diffusion-policy backbone, LIBERO simulator, and the
DSRL-NA two-critic SAC stack on top of jaxrl2.

This guide produces a single uv-managed Python venv (`.venv/`) that the
project's `uv run` and `bash examples/scripts/run_libero_robometer_na.sh`
launchers expect.

## Prerequisites

| | Tested |
|---|---|
| OS | Linux (kernel 6.x), x86_64 |
| Python | 3.11 (CPython) |
| GPU | NVIDIA, compute capability ≥ 8.0; verified on RTX 5090 (cc 12.0) |
| Driver | NVIDIA driver supporting CUDA 12.x |
| Disk | ~30 GB for the venv + cached pi0 checkpoints (more for buffer/logs) |
| RAM | 60 GB+ (replay-buffer K/V cache is host-resident at ~18 MB/slot) |
| Tools | `uv`, `git`, `git-lfs` (optional), `ssh` access to the project's GitHub orgs |

Install `uv` if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1. Clone the repository (with submodules)

The project pulls openpi and LIBERO as git submodules:

```bash
git clone --recurse-submodules git@github.com:RL-VLA/dsrl_pi0.git
cd dsrl_pi0

# If you forgot --recurse-submodules:
git submodule update --init --recursive
```

Submodule pointers (verify with `git submodule status`):

| Path | Repo | Pinned commit |
|------|------|---------------|
| `openpi/` | `git@github.com:arayabrain/openpi_pub.git` (also pushed to `RL-VLA/openpi`) | `5a703af` |
| `LIBERO/` | `git@github.com:nakamotoo/LIBERO.git` | (per `.gitmodules`) |

## 2. Create the venv and install dependencies

There are two supported ecosystem stacks. **Choose one.**

### Option A — Newer ecosystem with our forward-compat shims (default for this fork)

This is what the fork's running setup uses. Runs jax 0.5.3 with newer orbax /
jaxtyping; relies on the four guarded shims in `openpi/src/openpi/...` (already
on this branch).

```bash
uv venv .venv --python 3.11

# jax CUDA, project deps, openpi (editable), LIBERO (editable)
UV_LINK_MODE=copy uv pip install --python .venv/bin/python \
    "jax[cuda12]==0.5.3" \
    -e openpi \
    -e openpi/packages/openpi-client \
    -e LIBERO \
    -e .

# Required helpers
UV_LINK_MODE=copy uv pip install --python .venv/bin/python \
    pytest chex \
    "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu

# Newer transitives that need our shims
UV_LINK_MODE=copy uv pip install --python .venv/bin/python --upgrade \
    "orbax-checkpoint==0.11.36" \
    "jaxtyping>=0.3"
```

### Option B — Strict upstream-pinned ecosystem (no shims would be needed)

Matches the stack openpi's `pyproject.toml` declares (jaxtyping 0.2.36,
orbax-checkpoint 0.11.13, ml-dtypes 0.4.1, tensorstore 0.1.74, numpy<2). The
forward-compat shims in our openpi fork are no-ops here (verified via
`hasattr` / `getattr` guards).

```bash
uv venv .venv --python 3.11
cd openpi
UV_LINK_MODE=copy uv sync --python ../.venv/bin/python --frozen
cd ..

# After sync, force the trio to match the pin (uv sync sometimes pulls newer jaxlib)
UV_LINK_MODE=copy uv pip install --python .venv/bin/python --reinstall \
    "jax==0.5.3" "jaxlib==0.5.3" "jax-cuda12-plugin==0.5.3" "jax-cuda12-pjrt==0.5.3"

# Apply uv overrides as separate pins (uv pip install ignores override-deps)
UV_LINK_MODE=copy uv pip install --python .venv/bin/python --reinstall \
    "numpy==1.26.4" "tensorstore==0.1.74" "ml-dtypes==0.4.1"

# LIBERO + jaxrl2
UV_LINK_MODE=copy uv pip install --python .venv/bin/python -e LIBERO -e .

# Project-level extras
UV_LINK_MODE=copy uv pip install --python .venv/bin/python pytest chex \
    "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu
```

### Verify the installation

```bash
.venv/bin/python -c "
import jax, jaxlib, orbax.checkpoint as o
print(f'jax={jax.__version__} jaxlib={jaxlib.__version__} orbax={o.__version__}')
print('devices:', jax.devices())
"
```

Expected output ends with `devices: [CudaDevice(id=0), ...]` and `platform: gpu`
on a small matmul.

## 3. Cache the pi0 checkpoints

The training scripts pull these from a public GCS bucket on first use and cache
under `$OPENPI_DATA_HOME/openpi-assets/checkpoints/`. To pre-cache (≈ 26 GB
total for both):

```bash
# pi0_libero (~14 GB) — required for any libero run
.venv/bin/python -c "
import gcsfs, time
fs = gcsfs.GCSFileSystem(token='anon')
t0 = time.time()
fs.get('openpi-assets/checkpoints/pi0_libero/',
       'openpi/openpi-assets/checkpoints/pi0_libero/', recursive=True)
print(f'done in {time.time()-t0:.1f}s')
"
touch -d "2025-03-01" openpi/openpi-assets/checkpoints/pi0_libero  # marks cache as fresh

# pi0_aloha_sim (~12 GB) — only needed if running aloha env
.venv/bin/python -c "
import gcsfs, time
fs = gcsfs.GCSFileSystem(token='anon')
t0 = time.time()
fs.get('openpi-assets/checkpoints/pi0_aloha_sim/',
       'openpi/openpi-assets/checkpoints/pi0_aloha_sim/', recursive=True)
print(f'done in {time.time()-t0:.1f}s')
"
touch -d "2025-03-01" openpi/openpi-assets/checkpoints/pi0_aloha_sim
```

## 4. (Optional) Robometer reward server

The `run_libero_robometer*.sh` scripts call out to a Robometer FastAPI server
for dense rewards. If you don't have one running, set
`ROBOMETER_REWARD_KIND=libero_sparse` to fall back to env-only rewards.

To run the server (use a separate venv from your robometer checkout):

```bash
# server side — adjust paths to your robometer checkout / venv
<robometer_repo>/.venv/bin/python <robometer_repo>/robometer/evals/eval_server.py \
    server_url=0.0.0.0 server_port=8000
# verify:
curl http://localhost:8000/health
# {"status":"healthy","available_gpus":1,"total_gpus":1}
```

The server needs ~14 GB GPU memory; pin it to a different GPU than your
training run (e.g., GPU 0 for robometer, GPU 1 for training).

## 5. First training run

Smoke test (≈ 5 min after warmup, batch=128, traces enabled):

```bash
DEVICE_ID=1 bash examples/scripts/run_libero_robometer_na_speedtest.sh
```

Production run (≈ 5 days for 40k outer SAC steps with reference inner-step
counts):

```bash
DEVICE_ID=1 bash examples/scripts/run_libero_robometer_na.sh
```

Checkpoints land in `./logs/<wandb_project>/<expname>/checkpoint_*`. Wandb
runs are saved locally under `/tmp/tmp*/<expname>/wandb/` and synced if
`WANDB_MODE` isn't `disabled`.

## 6. Test suite

```bash
PYTHONPATH=. uv run python -m pytest tests/ -q
```

Currently 17 tests cover the SAC action-shape lift and DSRL-NA replay buffer
schema. All should pass on a freshly set up venv.

## Common pitfalls

| Symptom | Cause / fix |
|---------|-------------|
| `negative dimensions are not allowed` in DummyEnv | `--resize_image` defaulted to `-1`. Make sure your shell heredoc doesn't break the `\`-continued `uv run` line with an inline `# comment` (bash treats it as end-of-command). Move comments to *before* the `uv run`. |
| `JAX plugin jax_cuda12_plugin version 0.5.3 is installed, but it is not compatible with the installed jaxlib version 0.10.0` | Some transitive dep upgraded jaxlib past the lockfile pin. Force-reinstall the trio: `uv pip install --reinstall "jax==0.5.3" "jaxlib==0.5.3" "jax-cuda12-plugin==0.5.3" "jax-cuda12-pjrt==0.5.3"`. |
| `jax.experimental.layout has no attribute DeviceLocalLayout` | orbax-checkpoint version too old for installed jax. Either upgrade to `0.11.36` or apply uv overrides for `tensorstore==0.1.74`, `ml-dtypes==0.4.1`, `numpy<2` to keep 0.11.13. |
| `RESOURCE_EXHAUSTED: Out of memory while trying to allocate ...` during NA training | K/V cache filled host RAM. Lower `--online_buffer_size` (default 1000 ≈ 18 GB at 18 MB/slot) or set `--save_kv_cache 0`. See `docs/dsrl_na_cli_args.md`. |
| `cuDNN ... compute capability 12.0 ... will be jit-compiled from PTX` | Benign on the 5090. The cuDNN bundle in jax 0.5.3 supports cc 12.0; the warning is from one bundled kernel that PTX-compiles on first call. After warmup, perf is normal (~68 ms/pi0 inference verified). |

## Reference

- `docs/dsrl_na_cli_args.md` — every CLI flag for the launcher with semantics.
- `docs/dsrl_na_reference_hyperparams.md` — reference values from `run_aloha_sim_na.sh`.
- `docs/dsrl_na_speed_comparison_v2.md` — head-to-head timings vs the arayabrain reference.
- `docs/training_loop.md` — high-level training loop walkthrough.
- `tasks/` — implementation plans (action lift, DSRL-NA integration, scattered samples).
