# DSRL-on-pi0 Training Loop

This document walks through what `examples/scripts/run_libero.sh` (or `run_aloha.sh`) actually executes, step by step, so that the config-param reference in [`config_params.md`](./config_params.md) makes sense.

The pipeline:

```
run_libero.sh  →  examples/launch_train_sim.py  →  examples/train_sim.py::main()
                                                       └→ examples/train_utils_sim.py::trajwise_alternating_training_loop()
```

## 1. What "DSRL on pi0" means

- `pi0` is a frozen diffusion policy (`openpi` checkpoint, e.g. `pi0_libero` or `pi0_aloha_sim`). It takes images + robot state + a language prompt and denoises a Gaussian noise chunk of shape `(50, 32)` into a chunk of 50 actions. The **32** is the model's noise / action-space dimension; the **50** is the model's action horizon.
- A SAC agent (`PixelSACLearner`) learns to **steer pi0's sampling noise**. Instead of sampling noise from `N(0, I)`, the critic/actor pick a noise chunk that makes pi0 produce better actions.
- SAC only steers the **first `query_freq` entries** of the 50-step noise chunk — the remaining `50 - query_freq` entries are filled by repeating the last SAC-predicted entry. The env only executes those first `query_freq` actions before pi0 is re-queried.

So the SAC *action* is a tensor of shape `(query_freq, 32)`, flattened to `query_freq * 32`. For libero with `query_freq=20`, that is a 640-dim continuous action.

## 2. Boot sequence (`main` in `train_sim.py`)

1. **Devices & sharding.** Builds a `jax.sharding.Mesh(devices, ('batch',))` and a `NamedSharding(..., PartitionSpec('batch'))`. `shard_fn` pushes every batch onto devices along the leading (batch) axis.
2. **Output dir.** Creates `$EXP/<expname>` where `<expname>` is `<prefix>_<seed>[...]` (from `create_exp_name`).
3. **Env.**
   - `libero`: `libero_90` benchmark, hardcoded `task_id = 57`, rendered at `256×256`, `max_timesteps = 400`, `env_max_reward = 1`.
   - `aloha_cube`: registers `gym_aloha/AlohaTransferCube-v0` with `max_episode_steps=400`, obs type `pixels_agent_pos`, `env_max_reward = 4`, `max_timesteps = 400`.
4. **WandB logger.** Creates one run grouped by `<prefix>_<launch_group_id>`.
5. **DummyEnv.** A stub `gym` env whose action space is `Box((1, 32))` (pi0 noise) and observation space is `{pixels: (H, H, 3·num_cameras, 1), state?: (state_dim, 1)}`. It's only used to compute observation/action shapes for the SAC networks.
6. **pi0 policy.** Loads `openpi_config.get_config("pi0_libero"|"pi0_aloha_sim")` and the corresponding `s3://openpi-assets/checkpoints/...` checkpoint into `agent_dp` via `policy_config.create_trained_policy`.
7. **SAC agent.** `agent = PixelSACLearner(seed, sample_obs, sample_action, **train_kwargs)`. `sample_action.shape = (1, 1, 32)`, so `agent.action_chunk_shape = (1, 32)` and `agent.action_dim = 32` — but see §6 below; `action_chunk_shape` gets overwritten indirectly via the collected trajectory's action tensor.

   Actually the action seen by SAC is shape `(query_freq, 32)`. The constructor uses the `DummyEnv` action (shape `(1, 32)`) only to size the networks. When `agent.sample_actions(obs_dict)` is called in the rollout, it produces a `query_freq * 32` flat action, which `collect_traj` reshapes to `(query_freq, 32)` using `agent.action_chunk_shape` — this is set once at `PixelSACLearner.__init__` from `actions.shape[-2:]`. **Sanity check required when changing `--query_freq`**: look at `pixel_sac_learner.py:137` to confirm the action spec matches.
8. **Replay buffer.** `ReplayBuffer(obs_space, action_space, capacity = max_steps // multi_grad_step)`. Important: `ReplayBuffer.insert` auto-doubles capacity on overflow (`replay_buffer.py:116`), so this is only an initial allocation, not a hard cap.
9. Hands control to `trajwise_alternating_training_loop(...)`.

## 3. Outer loop: alternate rollout and SAC updates

`trajwise_alternating_training_loop` (in `examples/train_utils_sim.py:91`):

```python
replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
if shard_fn is not None:
    replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

while i <= variant.max_steps:                 # i = gradient-step counter
    traj = collect_traj(variant, agent, env, i, agent_dp)   # one full episode
    add_online_data_to_buffer(variant, traj, online_replay_buffer)

    num_gradsteps = (variant.num_online_gradsteps_batch
                     if variant.get("num_online_gradsteps_batch", -1) > 0
                     else len(traj["rewards"]) * variant.multi_grad_step)

    if len(online_replay_buffer) > variant.start_online_updates:
        for _ in range(num_gradsteps):
            batch = next(replay_buffer_iterator)     # size = variant.batch_size
            update_info = agent.update(batch)        # one SAC gradient step
            i += 1
            # periodic log / eval / checkpoint
```

Key facts:

- **The terminating counter `i` is gradient steps, not env steps.** `max_steps` = total SAC updates, not env interactions.
- **UTD (updates-to-data) = `multi_grad_step`.** After an episode with `N` replay-buffer insertions (i.e. `N` pi0 queries), the loop runs `N * multi_grad_step` gradient steps before collecting the next episode. `N ≈ max_timesteps / query_freq` — for libero, 400/20 = 20, so each episode yields ~20 * 20 = 400 gradient steps.
- **Warm-up.** Gradient updates only start once `len(online_replay_buffer) > start_online_updates`. Before then, episodes are collected using standard Gaussian noise (`i == 0` branch in `collect_traj`).
- **Logging.** Every `log_interval` gradient steps, scalar and histogram training metrics are logged. Every `eval_interval` gradient steps, `perform_control_eval` runs `eval_episodes` rollouts and logs videos. If `checkpoint_interval != -1`, the SAC agent is saved every that many steps.

## 4. Rollout: `collect_traj`

For each env step `t in [0, max_timesteps)`:

1. Build `obs_dict` for SAC from the resized (default `64×64`) agentview image, optionally concatenated with proprio state.
2. **Only every `query_freq` steps**, re-sample a fresh noise chunk and denoise with pi0:
   - If `i == 0` (i.e. SAC hasn't been updated yet): `noise ~ N(0, I)` of shape `(1, query_freq, 32)`, padded to 50 by repeating the last entry. This is the pure base-policy rollout.
   - Otherwise: `actions_noise = agent.sample_actions(obs_dict)` (shape `(query_freq * 32,)`), reshape to `(query_freq, 32)`, pad to 50 by repeating the last row. This is SAC's noise.
   - Feed to `agent_dp.infer(obs_pi_zero, noise=noise)["actions"]` → 50-step action chunk.
3. Execute `actions[t % query_freq]` in the environment. Record raw rewards and images.
4. Append the SAC-sized `actions_noise` to `action_list` (one entry per pi0 query, not per env step).

After the episode:

- **`episode_return` for logging** is the *sum of raw env rewards*.
- **Sparse SAC reward relabeling** (`train_utils_sim.py:278`):
  - `is_success = (last reward == env_max_reward)`
  - If success: `rewards = [-1, -1, ..., -1, 0]` of length `query_steps` (pi0-query count), `masks = [1, 1, ..., 1, 0]`.
  - Else: `rewards = [-1] * query_steps`, `masks = [1] * query_steps`.
  - So every SAC transition carries a constant −1 except the terminal success step (which is 0, mask 0). This is the "−1 until success" sparse-reward scheme.

## 5. Replay-buffer insertion (`add_online_data_to_buffer`)

One *pi0 query* is one *SAC transition*. For each `t in [0, query_steps)`:

```python
insert_dict = dict(
    observations   = obs_list[t],            # (H, H, 3·num_cameras, 1)
    next_observations = obs_list[t+1],
    actions        = actions[t],             # shape (query_freq, 32)
    next_actions   = actions[t+1]  (or actions[t] if t is last),
    rewards        = rewards[t],             # -1 or 0
    masks          = masks[t],               # 1 or 0
    discount       = variant.discount ** discount_horizon,   # γ^query_freq
)
online_replay_buffer.insert(insert_dict)
online_replay_buffer.increment_traj_counter()
```

Two important consequences:

- **The stored discount is `γ^query_freq` (= `γ^discount_horizon`).** Bellman targets use this per-chunk discount, so `--discount` is the per-env-step γ and the loop handles chunking automatically.
- **`masks=0` only on the successful terminal step.** Time-out truncations don't set mask=0 — the bootstrap still flows through, which is usually what you want for episodic tasks with a fixed horizon.

## 6. SAC update step

`agent.update(batch)` in `PixelSACLearner` (jit-compiled `_update_jit` in `pixel_sac_learner.py:42`):

1. **Image augmentation.** `batched_random_crop` on `batch["observations"]["pixels"]`; if `color_jitter=True` also `color_transform`. Same for next-obs if `aug_next=True`.
2. **Critic update** (`update_critic`): REDQ-style ensemble of `num_qs` Q-networks. `critic_reduction='mean'` uses the mean over ensemble for target; `'min'` is clipped double-Q style.
3. **Target critic** (`soft_target_update` with Polyak `tau`).
4. **Actor update** (`update_actor`): tanh-squashed Gaussian policy `LearnedStdTanhNormalPolicy`, squashed to `[-action_magnitude, action_magnitude]`.
5. **Temperature update** (`update_temperature`): targets `target_entropy`. Default `'auto'` → `-action_dim / 2` (so for libero with `query_freq=20`, action_dim = 640, target_entropy = −320). `run_aloha.sh` overrides with `--target_entropy 0.0`.

All three networks share a `PixelMultiplexer` visual encoder (configurable via `encoder_type`, defaults to the small 4-layer CNN from `encoder_type='small'` in both libero and aloha scripts).

## 7. Eval: `perform_control_eval`

Runs `eval_episodes` rollouts with a fresh RNG (`PRNGKey(seed + 456)`), each up to `max_timesteps` steps. For each rollout:

- Same pi0 query pattern as in `collect_traj`, except:
  - `i == 0` always samples from `N(0, I)` (evaluates the *base* policy).
  - Otherwise, `agent.sample_actions` is used (deterministic or stochastic depending on agent internals; here it's stochastic but uses the mean of the tanh-Gaussian at eval time via `agent.sample_actions` — check `Agent.sample_actions` if you need to change this).
- Logs video `eval_video/{rollout_id}` to wandb (fps=50).
- Aggregates `evaluation/success_rate`, `evaluation/avg_return`, `evaluation/avg_episode_len`, and `evaluation/Reward >= r` for `r = 0..env_max_reward`.

## 8. Gradient-step budget example (libero defaults)

```
max_timesteps  = 400
query_freq     = 20          →  query_steps per episode = 20
multi_grad_step= 20          →  gradient steps per episode = 20 * 20 = 400
max_steps      = 500_000     →  total episodes ≈ 500_000 / 400 = 1_250
                              →  total env steps ≈ 1_250 * 400 = 500_000
start_online_updates = 500   →  ~25 warm-up episodes (each adds 20 transitions)
eval_interval  = 10_000      →  ~25 evals over the run
batch_size     = 256         →  each SAC update sees 256 pi0-query transitions
```

## 9. Shapes quick-reference

| Tensor | Shape | Where |
|---|---|---|
| pi0 denoised action chunk | `(50, action_dim)` | `agent_dp.infer(...)["actions"]` |
| pi0 noise input | `(1, 50, 32)` | `agent_dp.infer(..., noise=...)` |
| SAC observation image | `(B, resize_image, resize_image, 3·num_cameras, 1)` | replay buffer |
| SAC state (if `add_states`) | `(B, state_dim, 1)` | state_dim = 8 (libero) / 14 (aloha) |
| SAC action | `(B, query_freq, 32)` | replay buffer `actions` |
| SAC actor output (flat) | `(B, query_freq * 32)` | `LearnedStdTanhNormalPolicy` |
| Critic ensemble output | `(num_qs, B)` | `StateActionEnsemble` |

## 10. Relevant source files

- `examples/launch_train_sim.py` — CLI argparse + `train_args_dict`.
- `examples/train_sim.py::main` — env / pi0 / SAC / replay-buffer construction.
- `examples/train_utils_sim.py::trajwise_alternating_training_loop` — outer loop.
- `examples/train_utils_sim.py::collect_traj` — rollout and reward relabeling.
- `examples/train_utils_sim.py::add_online_data_to_buffer` — transition insertion (`γ^query_freq` discount).
- `examples/train_utils_sim.py::perform_control_eval` — eval.
- `jaxrl2/agents/pixel_sac/pixel_sac_learner.py` — SAC networks and `_update_jit`.
- `jaxrl2/data/replay_buffer.py` — auto-doubling replay buffer.
- `jaxrl2/utils/launch_util.py::parse_training_args` — auto-registers `train_args_dict` entries as CLI flags.
