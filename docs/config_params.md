# Configurable Parameters (LIBERO defaults)

Everything that controls a training run for DSRL-on-pi0. Read [`training_loop.md`](./training_loop.md) first for context — this doc is the knob-by-knob reference.

**Default column** shows the value the run gets when `examples/scripts/run_libero.sh` is executed unmodified. Cells in *italic* are "not passed on the CLI" → the script-embedded argparse default applies.

**Source of defaults**:
- CLI argparse defaults: `examples/launch_train_sim.py:10-27`.
- `train_kwargs` defaults: `examples/launch_train_sim.py:29-52` (`train_args_dict`).
- Script overrides: `examples/scripts/run_libero.sh`.
- Env-dependent constants (hardcoded, not CLI-overridable): `examples/train_sim.py:110-133`.

To override a CLI flag, add `--<name> <value>` to the run script (the `uv run examples/launch_train_sim.py \ ...` block). To override a non-CLI constant, edit the listed source file.

---

## 1. Run-identity and I/O

| Flag | Type | LIBERO default | Description | Code |
|---|---|---|---|---|
| `--seed` | int | `0` | PRNG seed for SAC, env `env.seed(seed)`, and eval RNG (`seed+456`). | `launch_train_sim.py:10` |
| `--launch_group_id` | str | *''* | WandB group suffix (`<prefix>_<launch_group_id>`). | `launch_train_sim.py:11` |
| `--env` | str | `libero` | `libero` or `aloha_cube`. Switches pi0 checkpoint, env construction, env-specific constants. | `launch_train_sim.py:13` |
| `--prefix` | str | `dsrl_pi0_libero` | Run-name prefix. If empty, auto-assigned from `uuid`. | `launch_train_sim.py:23` |
| `--suffix` | str | *''* | Optional run-name suffix. | `launch_train_sim.py:24` |
| `--wandb_project` | str | `DSRL_pi0_Libero` | WandB project (set via `$proj_name` in shell). | `launch_train_sim.py:20` |
| `--algorithm` | str | `pixel_sac` | Only `pixel_sac` is wired up; switching requires code changes. | `launch_train_sim.py:22` |
| `WANDB_ENTITY` (env var) | str | `peterchenyipu` | WandB entity. Set in `run_libero.sh`. | `run_libero.sh:15` |
| `$EXP` (env var) | path | `./logs/DSRL_pi0_Libero` | Output dir root. The run writes to `$EXP/<prefix>_<seed>[_suffix]`. | `train_sim.py:104` |
| `OPENPI_DATA_HOME` (env var) | path | `./openpi` | Where `openpi` caches the pi0 checkpoint after S3 download. | `run_libero.sh:10` |

## 2. Training schedule / budget

| Flag | Type | LIBERO default | Description |
|---|---|---|---|
| `--max_steps` | int | `500000` | **Total SAC gradient-step budget.** Loop terminates when `i > max_steps`. Also sets initial replay-buffer capacity (`max_steps // multi_grad_step`), but the buffer auto-doubles on overflow. |
| `--multi_grad_step` | int | `20` | UTD ratio. After each episode, runs `len(traj["rewards"]) * multi_grad_step` gradient steps. With `query_freq=20` and `max_timesteps=400`, that's `20 * 20 = 400` gradient steps / episode. |
| `--start_online_updates` | int | `500` | SAC `update()` is skipped until `len(online_replay_buffer) > start_online_updates`. Episodes are still collected and logged; rollouts before this point use `N(0, I)` noise (pure base pi0). |
| `--batch_size` | int | `256` | SAC minibatch. Must be divisible by `len(jax.local_devices())`. |
| `--log_interval` | int | `500` | Scalar/histogram WandB log every N gradient steps. |
| `--eval_interval` | int | `10000` | Runs `perform_control_eval` every N gradient steps (also logs at the end-of-episode boundary containing `i % eval_interval == 0`). |
| `--eval_episodes` | int | `10` | Rollouts per eval call. |
| `--checkpoint_interval` | int | *`-1`* | Disabled. Set positive to dump SAC state via `agent.save_checkpoint` every N steps. |
| `--num_online_gradsteps_batch` | int | *not registered* | Overrides the UTD formula if set > 0: forces a fixed `num_gradsteps` per episode. Not a CLI flag — accessed via `variant.get("num_online_gradsteps_batch", -1)`, so you'd have to inject it manually. | `train_utils_sim.py:113` |

## 3. Environment / rollout shape

These are **hardcoded per env** in `train_sim.py` — no CLI flag.

| Constant | LIBERO value | Aloha value | Where | Notes |
|---|---|---|---|---|
| `task_id` | `57` | n/a | `train_sim.py:113` | Index into `libero_90` task suite. Change by editing the line. |
| Rendering resolution | `256×256` | *rgb_array* | `train_sim.py:115` / `:129` | `camera_heights/widths=256` for libero. Resized to `resize_image` for SAC inputs. |
| `max_timesteps` | `400` | `400` | `train_sim.py:119` / `:126` / `:132` | Per-episode env step cap. Libero ends early on `done`. Aloha is capped by `gym.register(..., max_episode_steps=400)`. |
| `env_max_reward` | `1` | `4` | `train_sim.py:118` / `:131` | Final reward that counts as success (used by `is_success = reward == env_max_reward`). |
| `state_dim` (proprio) | `8` | `14` | `train_sim.py:69-72` | Used only if `add_states=1`. |

### Rollout control

| Flag | Type | LIBERO default | Description |
|---|---|---|---|
| `--query_freq` | int | `20` | **pi0 query period AND SAC action chunk length.** Every `query_freq` env steps, a fresh noise chunk is drawn and passed to pi0. SAC action shape = `(query_freq, 32)`. Also used in the per-chunk discount: `stored_discount = discount ** query_freq`. |
| `--resize_image` | int | `64` | SAC input image size. `-1` disables resizing (uses raw `256×256` — unlikely to be useful). |
| `--add_states` | int (bool) | *`1`* | Append proprio state to SAC obs dict. |

### pi0 noise / action space (hardcoded)

| Constant | Value | Where | Notes |
|---|---|---|---|
| pi0 action horizon | `50` | `train_utils_sim.py:234, 241` | Noise is always padded to `(1, 50, 32)` by repeating the last SAC-predicted row. |
| pi0 action / noise dim | `32` | `train_sim.py:74` | DummyEnv `action_space = Box((1, 32))`. Changing this requires a different pi0 checkpoint. |

## 4. SAC hyperparameters (`train_args_dict`)

All entries in `train_args_dict` are auto-registered as CLI flags by `parse_training_args` (`jaxrl2/utils/launch_util.py:3`), so override as e.g. `--actor_lr 5e-5`. They end up in `variant["train_kwargs"]` and are passed directly as `**kwargs` to `PixelSACLearner.__init__`.

| Flag | Type | LIBERO default | Description |
|---|---|---|---|
| `--actor_lr` | float | *`1e-4`* | Adam LR for actor. |
| `--critic_lr` | float | *`3e-4`* | Adam LR for critic ensemble. |
| `--temp_lr` | float | *`3e-4`* | Adam LR for SAC temperature. |
| `--discount` | float | `0.999` | **Per-env-step γ.** Stored discount per replay transition = `γ ** query_freq` (= `0.999 ** 20 ≈ 0.980` for libero). |
| `--tau` | float | *`0.005`* | Polyak averaging coefficient for target critic. |
| `--hidden_dims` | int+ (tuple) | `128` (→ `(128,128,128)`) | Actor/critic MLP widths. If you pass a single value, `pixel_sac_learner.py:172` expands it to 3 layers. Pass multiple (`--hidden_dims 256 256`) for custom shape. |
| `--cnn_features` | int+ | *`(32,32,32,32)`* | Per-layer channel counts for `encoder_type='small'`. |
| `--cnn_strides` | int+ | *`(2,1,1,1)`* | Per-layer strides for the small CNN. |
| `--cnn_padding` | str | *`VALID`* | CNN padding. |
| `--latent_dim` | int | *`50`* | Dim of the `PixelMultiplexer` bottleneck between encoder and MLP. |
| `--critic_reduction` | str | *`mean`* | `'mean'` or `'min'` over the `num_qs`-ensemble in the Bellman target. |
| `--dropout_rate` | float | *`0.0`* | Actor MLP dropout (disabled by default). |
| `--aug_next` | int (bool) | *`1`* | Apply random-crop (and color-jitter if on) to next-obs pixels during critic update. |
| `--use_bottleneck` | bool→int | *`1`* | Enable the `latent_dim` bottleneck. |
| `--encoder_type` | str | *`small`* | One of `small, impala, impala_small, resnet_small, resnet_18_v1, resnet_34_v1, resnet_small_v2, resnet_18_v2, resnet_34_v2`. |
| `--encoder_norm` | str | *`group`* | `group` / `batch` / `layer` for ResNet-type encoders. |
| `--use_spatial_softmax` | bool→int | *`1`* | ResNet-only: spatial-softmax head. |
| `--softmax_temperature` | float | *`-1`* | Spatial-softmax temp (−1 = learnable). |
| `--target_entropy` | float / str | *`auto`* | `'auto'` → `-action_dim / 2` = `-(query_freq * 32) / 2` = `-320` for libero. For aloha the script overrides with `--target_entropy 0.0`. |
| `--num_qs` | int | *`10`* | Critic ensemble size (REDQ-style). |
| `--action_magnitude` | float | `1.0` | SAC action range `[-action_magnitude, action_magnitude]` (tanh-squashed). Aloha uses `2.0` (noise steering has wider range). |
| `--num_cameras` | int | *`1`* | Stack-count for multi-camera color-jitter. `>1` splits the channel axis into `num_cameras` groups before color-transforming each independently. |
| `--color_jitter` | *(hardcoded True)* | — | Constructor default, not exposed in `train_args_dict`. Edit `pixel_sac_learner.py:117` to disable. |
| `--init_temperature` | *(hardcoded 1.0)* | — | Same — not a CLI flag. |
| `--decay_steps` | int | *None* | Cosine LR decay for actor. Set via `--cosine_decay 1`: that flag is **popped** from `train_kwargs` in `train_sim.py:92` and, if truthy, sets `decay_steps = max_steps`. Note: `--cosine_decay` is not in `train_args_dict`, so you have to add it or pass it in a config. |

## 5. Environment variables

Set in `run_libero.sh` / `run_aloha.sh`:

| Var | LIBERO default | Purpose |
|---|---|---|
| `DISPLAY` | `:0` | X display (needed by some MuJoCo/EGL paths). |
| `MUJOCO_GL` | `egl` | MuJoCo renderer backend. |
| `PYOPENGL_PLATFORM` | `egl` | Forces EGL for PyOpenGL (libero only). |
| `MUJOCO_EGL_DEVICE_ID` | `$device_id` (0) | Which GPU MuJoCo uses for EGL. |
| `CUDA_VISIBLE_DEVICES` | `0` | JAX/SAC GPU. |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Don't grab all VRAM up front. |
| `PYTHONPATH` | `${PYTHONPATH:-}:.` | Makes `examples/*` importable when running from the repo root. |
| `OPENPI_DATA_HOME` | `./openpi` | Cache dir for `download.maybe_download(...)` pi0 checkpoints. |
| `EXP` | `./logs/DSRL_pi0_Libero` | Training output dir root. |
| `WANDB_ENTITY` | `peterchenyipu` | WandB entity; override to your own username/team. |

Also set in code (once):
- `XLA_FLAGS += --xla_gpu_triton_gemm_any=True` — `train_sim.py:5`.
- `compilation_cache.set_cache_dir($HOME/jax_compilation_cache)` — `train_sim.py:36`.

## 6. Full LIBERO invocation

For reference, the unmodified LIBERO CLI:

```bash
uv run examples/launch_train_sim.py \
  --algorithm pixel_sac \
  --env libero \
  --prefix dsrl_pi0_libero \
  --wandb_project DSRL_pi0_Libero \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --query_freq 20 \
  --hidden_dims 128
```

Everything else falls back to `argparse`/`train_args_dict` defaults listed above.

## 7. How to override — summary

| You want to change… | Do this |
|---|---|
| Any flag in §1/2/3-rollout/4 | Add `--<name> <value>` to the `uv run ...` block in the run script. |
| Env task, rendering res, `max_timesteps`, `env_max_reward` | Edit `examples/train_sim.py:110-133` — these are hardcoded per-env. |
| pi0 action horizon (50) or action-space dim (32) | Would need a different pi0 checkpoint; also update the `repeat`/`concatenate` in `train_utils_sim.py:234-245`. |
| Reward structure (−1/0 sparse) | Edit `train_utils_sim.py:278-287`. |
| `color_jitter` / `init_temperature` | Edit `PixelSACLearner.__init__` defaults in `jaxrl2/agents/pixel_sac/pixel_sac_learner.py:98-127`, or add them to `train_args_dict` in `examples/launch_train_sim.py`. |
| Replay-buffer behavior (e.g. cap, no auto-double) | Edit `jaxrl2/data/replay_buffer.py:116-144`. The initial capacity passed in is `max_steps // multi_grad_step` (`train_sim.py:158`). |
| SAC update internals (actor/critic/temp losses) | `jaxrl2/agents/pixel_sac/{actor,critic,temperature}_updater.py`. |
