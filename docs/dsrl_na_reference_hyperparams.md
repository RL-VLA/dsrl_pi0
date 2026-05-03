# Reference DSRL-NA Hyperparameters

Settings the arayabrain reference impl ships in
`/data3/dsrl_pi0_na/examples/scripts/run_aloha_sim_na.sh` — the only DSRL-NA
training script in their repo (no libero-NA variant). Use as the baseline
when comparing our impl's behaviour or speed.

## Device / serving
| Knob | Value | Notes |
|------|-------|-------|
| `device_id` | `0,1,2,3` | 4-GPU sharding by default |
| `local_policy_device` | `0,1,2,3` | pi0 sharded across 4 GPUs |
| `dsrl_device_idx` | `0` | which GPU the SAC learner lives on |
| `kv_cache_device` | `0` | overridden by `put_kv_cache_on_cpu` |
| `use_local_policy` | `1` | run pi0 in-process (vs websocket) |
| `save_kv_cache` | `1` | cache pi0 prompt-encoder K/V per slot |
| `put_kv_cache_on_cpu` | `1` | offload K/V to CPU; saves GPU mem |

## Env / action
| Knob | Value |
|------|-------|
| `env` | `aloha_cube` |
| `query_freq` | `50` |
| `max_timesteps` | `400` |
| `action_chunk_size` | `50` |
| `action_magnitude` | `0.75` |
| `resize_image` | `64` |
| `num_cameras` | `1` (default) |

## Training
| Knob | Value |
|------|-------|
| `batch_size` | `128` |
| `discount` | `0.999` |
| `seed` | `0` |
| `max_steps` | `1_000_000` |
| `multi_grad_step` | `20` (UTD knob; only consumed if `train_all_together=1`) |
| `num_initial_traj_collect` | `125` (warm-start the buffer with base-policy trajectories) |
| `online_buffer_size` | `150_000` |
| `num_qs` | `10` (critic ensemble size) |
| `hidden_dims` | `128` |
| `target_entropy` | `0.0` |

## Logging / eval
| Knob | Value |
|------|-------|
| `log_interval` | `100` |
| `eval_interval` | `5_000` |
| `eval_episodes` | `10` |
| `checkpoint_interval` | `10_000` |

## Pi0 / distillation
| Knob | Value |
|------|-------|
| `flow_integration_steps` | `10` |

## NA-specific structure
| Knob | Value | Notes |
|------|-------|-------|
| `action_critic_steps` | `15` | inner action-critic gradient steps per outer SAC step |
| `noise_critic_steps` | `5` | inner noise-critic distillation steps per outer SAC step |
| `noise_actor_steps` | `5` | inner actor steps per outer SAC step |
| `train_all_together` | `0` | when `0`, effective UTD = `action_critic_steps + noise_critic_steps = 20`; when `1`, UTD = `multi_grad_step` |

So **per outer SAC update they run 25 inner gradient steps**: 15 action-critic + 5 noise-critic + 5 actor. Our impl currently does 1 of each per outer step (`noise_critic_grad_steps=1`, all toggle flags default to `1`).

## Exploration / SAC bookkeeping
| Knob | Value | Effect |
|------|-------|--------|
| `grl_noise_sample` | `1` | distillation noise = 50/50 mix of fresh `N(0,I)` and actor-sampled noise |
| `backup_entropy` | `0` | no entropy bonus in the action-critic Bellman target |
| `noise_scale_inside` | `0` | bound actor distribution to `[-1, 1]`, then multiply by `action_magnitude` externally |

## Optional
| Knob | Value |
|------|-------|
| `restore_path` | `""` |
| `remote_host` / `remote_port` | `localhost:8090` (only used when `use_local_policy=0`) |

---

## Q&A

### What is "GRL" in `grl_noise_sample`?
The reference repo never spells out the acronym. From the CLI help and the
implementation in `examples/train_utils_na.py:choose_noise`:

> "For noise critic distillation, sample half the noise from the noise actor."

So it's a flag that controls the **distillation noise sampler**, not a reward
algorithm. When `grl_noise_sample=0`, distillation noise is always fresh
`N(0, I)`. When `1`, with probability `0.5` the noise is sampled from the
current actor instead of from the prior.

The behavioural intuition: once the actor has converged to a tight distribution,
distilling the noise critic only on Gaussian samples gives you a critic that
is accurate on **unreachable** latents and inaccurate where the actor
actually lives. Mixing in actor samples covers both regions. The "GRL"
prefix is most plausibly an internal abbreviation (e.g. for
"Generative-Replay-of-Latents" or similar) — without an explanatory comment
in the repo it should be treated as opaque project shorthand.

### Why does the actor update need a pi0 forward?
**It doesn't.** The actor update operates entirely against the **noise critic**
`Q_n(s, z)`. Concretely (`jaxrl2/agents/pixel_dsrl_na/actor_updater.py`):

```python
actions, log_probs = dist.sample_and_log_prob(seed=key)   # z ~ π(·|s)
actions = actions * noise_scale                           # external scale
qs = critic.apply_fn(..., observations, actions)          # Q_n(s, z)
actor_loss = (log_probs * temp - q).mean()                # SAC objective
```

No `agent_dp.infer_batch` call, no pi0 forward. The whole point of building
`Q_n` is so the actor can be trained without pi0 in the inner loop.

The pi0 calls per outer SAC step happen in **two other places**:

1. **Action-critic update** — needs pi0 to get `next_executed_actions`:
   `Q_a` is defined over diffused env-actions, so the Bellman target
   `r + γ * Q_a_target(s', pi0(s', z'))` requires running pi0 on `s'`
   with the next-step noise sampled from the actor.

2. **Noise-critic distillation** — needs pi0 to build the
   `(z_distill, pi0(s, z_distill))` training pair:
   `MSE(Q_n(s, z_distill), Q_a(s, pi0(s, z_distill)))`. The pi0 call here
   produces the `distill_actions` that the action critic is then queried on.

Both are batched, JIT-warm, and (with K/V cache on) cheap. They produce the
inputs that the JITted updaters consume; the actor updater itself is pi0-free.

### What does the K/V cache logic mean? Is the cache saved during the action-critic / noise-critic updates?

**No — the K/V cache is saved at rollout time, not during updates.** Updates
*read* the cache.

#### What is cached
Pi0 splits naturally into two stages:

1. **Prompt encoder** — runs on the (image, state, language-prompt) tuple
   and produces K/V tensors used by the diffusion transformer for cross-
   attention. This part is *deterministic in the observation* — it doesn't
   depend on the noise.
2. **Diffusion stack** — runs `flow_integration_steps` denoising iterations
   over the action latents, conditioned on the prompt-encoder K/V from
   step 1. This part *does* depend on the noise input.

For a given observation `s`, the K/V from step 1 is the same no matter what
noise `z` you decode with. So you can cache it once and reuse it.

#### When the cache is populated
At rollout time (`collect_traj`):

```python
if save_kv_cache:
    k_cache, v_cache = agent_dp.get_prefix_rep_and_kv_cache(obs_pi_zero)
    online_replay_buffer.cache_storage.insert(k_cache, v_cache, slot_idx)
```

The K/V is computed once per chunk-start observation and stored in the
buffer's `CacheStorage` keyed by the slot index. The chunk's pi0 forward
that produced the executed action also uses this K/V, so there's no
duplicated work.

#### When the cache is read
During each outer SAC update step, two pi0 forwards happen on batches
sampled from the replay buffer (in `_na_update_step` in
`examples/train_utils_sim_robometer.py`):

1. **Action critic** — for `next_observations` in the batch:
   ```python
   batch_dict['original_next_k_cache'] = online_replay_buffer.get_cache(next_indices, 'k')
   batch_dict['original_next_v_cache'] = online_replay_buffer.get_cache(next_indices, 'v')
   agent_dp_next, _ = get_next_actions_from_dp(agent_dp, batch_dict, next_noise, ...)
   ```
   Pi0 receives `kv_cache=(k, v)` and skips the prompt-encoder forward.

2. **Noise-critic distillation** — for current `observations` in the batch:
   ```python
   distill_input['original_k_cache'] = online_replay_buffer.get_cache(indices, 'k')
   distill_input['original_v_cache'] = online_replay_buffer.get_cache(indices, 'v')
   distill_noise, distill_actions, _ = generate_distillation_batch(distill_input, ...)
   ```
   Same trick — pi0 reuses the cached K/V for the distillation pair.

So per outer SAC update there are **two pi0 forwards** (one for action-critic
target, one for distillation pair), both reading cached K/V. The updates
themselves (action-critic gradient step, noise-actor gradient step,
noise-critic distillation gradient step) do **not** call pi0 and do **not**
modify the cache.

#### Where the cache lives
With `put_kv_cache_on_cpu=1` (the reference default), the cache lives in
host memory and is paged onto the GPU only when pi0 needs it for a forward.
That's because at `online_buffer_size=150_000` and the per-slot K/V being
many MB, GPU memory would otherwise be saturated. Our impl currently keeps
the cache on the same device as the buffer.

#### Cost amortization
Without the cache, every pi0 forward at update time re-runs the prompt
encoder, which is the dominant pi0 cost on long prompts. With the cache,
update-time pi0 ≈ diffusion stack only (`flow_integration_steps`
iterations of a much smaller transformer). On the reference's 4-GPU,
batch=128, 25-inner-steps configuration, this is the difference between
"trainable" and "intractable".
