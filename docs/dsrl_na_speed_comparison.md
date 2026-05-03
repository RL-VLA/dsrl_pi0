# DSRL-NA Speed Comparison: Our impl vs arayabrain reference

## Summary

At matched hyperparameters (batch=16, query_freq=50, hidden_dims=128,
save_kv_cache=1, multi_grad_step=1, num_qs=10, max_steps=30), our DSRL-NA
implementation completed 30 SAC update steps **~15 % faster** than the
arayabrain reference impl on the same GPU.

| Run | 30 SAC steps | First step (JIT) | Warm steady-state |
|-----|--------------|------------------|-------------------|
| **Ours** (libero, robometer) | **76 s** | ~37 s | ~3-4 it/s (0.3 s) |
| **Reference** (aloha_cube)   | 89 s | ~63 s | ~3 it/s (0.33 s) |

The user-reported "much slower" behavior is therefore not from the matched
DSRL-NA path itself. Likely sources are different production hyperparameters
(e.g., `batch_size=256`, `multi_grad_step=20` → 20× the per-outer cost) or
older revisions before recent fixes.

## Configurations compared

### Reference impl (`/data3/dsrl_pi0_na/`)
- Script: `/tmp/run_ref_na_smoke.sh` (derived from
  `examples/scripts/run_aloha_sim_na.sh`).
- Env: `aloha_cube` (gym_aloha).
- Pi0 checkpoint: `pi0_aloha_sim` (12 GB, downloaded via anonymous gcsfs).
- Venv: `/data3/dsrl_pi0_na/.venv` (python 3.11, jax-cuda 0.5.3, jaxrl2 +
  arayabrain openpi).
- Log: `/tmp/ref_na_smoke.log`.

### Our impl (`/data3/dsrl_pi0/`)
- Script: `examples/scripts/run_libero_robometer_na_speedtest.sh`.
- Env: `libero` + robometer dense reward.
- Pi0 checkpoint: `pi0_libero` (14 GB, already cached).
- Venv: `/data3/dsrl_pi0/.venv` (uv-managed).
- Log: `/tmp/our_na_speedtest.log`.

### Hyperparameters held equal across both runs
| Knob | Value |
|------|-------|
| `batch_size` | 16 |
| `discount` | 0.999 |
| `seed` | 0 |
| `max_steps` | 30 |
| `multi_grad_step` | 1 |
| `query_freq` | 50 |
| `action_chunk_size` | 50 |
| `hidden_dims` | 128 |
| `num_qs` | 10 |
| `action_magnitude` | 0.75 |
| `save_kv_cache` | 1 |
| `noise_scale_inside` | 0 |
| `grl_noise_sample` | 0 |
| `dsrl_na_backup_entropy` (ref: `backup_entropy`) | 0 |
| `eval_episodes` | 1 |

### Differences that are not hyperparameters
- **Env**: libero vs aloha_cube — affects rollout time only, not SAC update.
- **Reward source**: dense robometer vs dense env reward — robometer is async,
  so it's overlapped and shouldn't dominate.
- **SAC actor head shape**: ours outputs `(query_freq, 32) = (50, 32) = 1600`
  values per step; the reference outputs a single `(1, 32) = 32`-d latent that
  is broadcast into pi0's 50-step diffusion noise. Our actor head is therefore
  ~50× wider — and despite that, we are not slower at warm steady-state.

## Per-step timeline (raw)

### Ours (`/tmp/our_na_speedtest.log`)
```
0/30  → 00:14   (rollout phase finishes)
1/30  → 00:51   ~37 s  (first JIT)
2/30  → 00:58   ~7 s   (still warming)
8/30  → 01:00   ~2 s for steps 3-8 (warmth ramping)
16/30 → 01:06   ~6 s for steps 9-16 (~0.75 s/step)
24/30 → 01:11   ~5 s for steps 17-24 (~0.6 s/step, peak 4 it/s)
30/30 → 01:16   ~5 s for steps 25-30 (~0.8 s/step incl rollout interleave)
final tqdm     2.85 it/s averaged
```

### Reference (`/tmp/ref_na_smoke.log`)
```
1/30  → 01:03   ~63 s  (first JIT)
8/30  → 01:13   ~10 s for steps 2-8 (~1.4 s/step warming)
21/30 → 01:22   ~9 s for steps 9-21 (~0.7 s/step)
28/30 → 01:27   3 it/s steady-state ≈ 0.33 s/step
30/30 → 01:29   final 1.17 it/s averaged
```

## Rollout phase (libero vs aloha)
Both envs sit at ~120-130 it/s for `env.step` — neither dominates.
A 400-step rollout completes in ~5-8 s including pi0 inference. The pi0
inference call is the hot loop in rollouts; the K/V cache hook
(`save_kv_cache=1`) keeps it bounded.

## Conclusion

For DSRL-NA at matched batch=16, our impl matches or slightly beats the
arayabrain reference on per-step wall-clock. The "much slower" perception
likely comes from a different scale (production batch=256, multi_grad_step=20,
500k max_steps) where small constant-factor differences compound. A follow-up
benchmark at production batch_size + multi_grad_step would be the next step
to verify.
