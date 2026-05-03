# DSRL-NA Speed Comparison v2 — production-scale, matched hyperparameters

Speed comparison after wiring the missing per-component step counts
(`action_critic_steps=15`, `noise_critic_steps=5`, `noise_actor_steps=5`),
`flow_integration_steps`, `online_buffer_size`, `num_initial_traj_collect`,
`target_entropy`, and `put_kv_cache_on_cpu` from the arayabrain reference into
our launcher and `_na_update_step`. Same machine, same GPU, same batch size,
all behavioural knobs aligned.

## Result (TL;DR)

At matched hyperparameters, **our impl is ~1.6× faster per gradient step**
than the arayabrain DSRL-NA reference.

| | Reference (`dsrl_pi0_na`, aloha_cube) | Ours (`dsrl_pi0`, libero + robometer) |
|---|---|---|
| Wall-clock for 200 SAC outer pbar steps | **3:20 (200 s)** | **33:19 (1999 s)** |
| Outer-pbar semantics | 1 outer = 1 inner gradient step | 1 outer = 25 inner gradient steps (15 ac + 5 nc + 5 na) |
| Total gradient steps in 200-outer run | ~200 | ~5000 |
| **Steady-state per inner gradient step** | **~0.60 s** | **~0.38 s** |
| Per outer SAC cycle (full 15 ac + 5 nc + 5 na) | ~12 s (ref's 20-iter cycle covering 15 ac + 5 nc; actor reuses iters from cycle 2) | ~9.55 s |
| First-iter JIT cost | ~63 s | ~30 s |

## Why ours is faster

The reference's inner loop interleaves three different jitted kernels by
iteration index inside one cycle:

```python
inter_step = grad_idx % 20
train_action_critic = (inter_step < 15)
train_noise_critic = (15 <= inter_step < 20)
train_noise_actor  = (inter_step < 5) and (cycle >= 1)
```

Each `inter_step` boundary triggers a JIT trace for the new toggle pattern,
plus inter-step Python work. Ours runs the three components in three
sequential blocks per outer SAC iter — three jitted kernels, then move on.
JAX caches each, so steady-state has no kernel-switching overhead; we also
fetch `next_indices` / `indices` K/V slices **once per outer step** instead
of once per inner iter (host→GPU paging is amortized).

## Run setup

### Hyperparameters held equal across both runs
| Knob | Value |
|------|-------|
| `batch_size` | 128 |
| `discount` | 0.999 |
| `seed` | 0 |
| `query_freq` | 20 (libero) / 50 (aloha) |
| `action_chunk_size` | 50 |
| `hidden_dims` | 128 |
| `num_qs` | 10 |
| `action_magnitude` | 0.75 |
| `target_entropy` | 0.0 |
| `save_kv_cache` | 1 |
| `put_kv_cache_on_cpu` | 1 |
| `grl_noise_sample` | 1 |
| `noise_scale_inside` | 0 |
| `dsrl_na_backup_entropy` (ref: `backup_entropy`) | 0 |
| `flow_integration_steps` | 10 |
| `online_buffer_size` | 150 000 |
| `num_initial_traj_collect` | 5 (truncated for speed run) |
| `action_critic_steps` | 15 |
| `noise_critic_steps` | 5 |
| `noise_actor_steps` | 5 |
| `max_steps` | 200 (outer pbar) |
| `multi_grad_step` (reference) | 20 |
| `multi_grad_step` (ours) | 1 (UTD driven by inner step counts) |
| `sac_action_chunk_size` (ours only) | 1 (matches reference's `(1, 32)` noise) |

### Differences that aren't hyperparameters
- **Env**: ref runs aloha_cube; ours runs libero + dense robometer reward. Both rollouts cap at ~120 it/s, so env-step cost is comparable.
- **Inner-loop scheduling order**: ref interleaves ac/nc/na inside a 20-step cycle; ours runs `[15× ac]` then `[5× nc]` then `[5× na]`. Same per-component step count per outer; different target-network freshness profile.

### Hardware / Software
- Single GPU (NVIDIA, bfloat16, jax-cuda 0.5.3, openpi commit `5a703af`).
- pi0_libero / pi0_aloha_sim checkpoints cached at `/data3/dsrl_pi0/openpi/openpi-assets/checkpoints/`.

## Raw timing extracts

### Reference (`/tmp/speed_ref.log`)
```
192/800 [02:57<06:04, 1.67 it/s]
193/800 [02:57<06:07, 1.65 it/s]
... (steady-state)
800/800 [09:03<00:00, 1.65 it/s]   # full inner loop completion
200/200 [03:20<00:00, 1.67 it/s]   # outer pbar exits at max_steps
```

### Ours (`/tmp/speed_ours.log`)
```
175/200 [29:12<03:58,  9.53 s/it]
176/200 [29:22<03:48,  9.52 s/it]
... (steady-state)
200/200 [33:19<00:00,  9.53 s/it]
```

## Run scripts

- `/tmp/speed_experiment.sh` — drives the whole experiment (Phase A: ref, Phase B: ours).
- `/tmp/run_ours_only.sh` — re-runs only Phase B after a code fix (used here).

## Caveats

- Both rollouts succeed at ~120 it/s, so env physics is not the bottleneck.
- The reference's interleaved scheduler may be intentionally pacing target
  updates differently from our sequential block scheduler. Speed ≠ learning
  quality — a longer run would be needed to confirm the two converge to
  similar returns.
- The "1.6×" headline is at this exact hyperparameter set. Doubling
  `noise_critic_steps` (more pi0 forwards) or doubling `multi_grad_step`
  would scale both implementations roughly linearly; the ratio shouldn't
  change much.

## Logs preserved
- `/tmp/speed_ref.log` (104 KB)
- `/tmp/speed_ours.log` (~?KB)
- `/tmp/speed_summary.txt` (phase markers)
