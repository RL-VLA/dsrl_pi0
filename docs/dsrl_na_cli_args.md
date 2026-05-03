# DSRL-NA CLI Arguments

Reference for the CLI flags added to `examples/launch_train_sim_robometer.py` for
the DSRL-NA two-critic variant. All flags labeled `[pixel_dsrl_na only]` are
ignored when `--algorithm pixel_sac`.

For algorithmic context see `docs/dsrl_na_integration_plan.md` and
`docs/sac_action_lift_plan.md`. For the wider robometer reward pipeline see
`docs/training_loop.md`.

## Algorithm selector

### `--algorithm {pixel_sac, pixel_dsrl_na}` (default: `pixel_sac`)

Picks the SAC variant the trainer instantiates.

- `pixel_sac` — original DSRL-SAC: a single noise critic `Q(s, z)` is trained
  directly on the noise latent `z` produced by the actor. The action critic
  bookkeeping below is unused.
- `pixel_dsrl_na` — DSRL-NA: two critics. The **action critic**
  `Q_a(s, a_executed)` is trained over the diffused env-actions pi0 produces
  from the noise latent. The **noise critic** `Q_n(s, z)` is then *distilled*
  from `Q_a` each update step by minimizing
  `MSE(Q_n(s, z), Q_a(s, pi0(s, z)))` on a fresh `(z, pi0(s, z))` pair. The
  actor is trained against `Q_n` exactly like in DSRL-SAC.

The DSRL-NA path additionally builds the round-robin `DSRLNAReplayBuffer`
(stores `original_observations`, optional pi0 K/V cache per slot) instead of
the standard buffer, and routes per-update SAC steps through `_na_update_step`
which calls pi0 on the sampled batch.

## Two-critic-specific knobs

### `--noise_critic_grad_steps INT` (default: `1`)

Reserved hook for "inner distillation steps per outer SAC update." The
arayabrain reference does 10 inner distillation steps for every 1 outer
critic+actor step. We currently always pass `1` through to the learner —
this flag is plumbed but not yet wired into the inner loop. Listed here so
the CLI surface matches the reference and the hook is discoverable for the
follow-up.

### `--dsrl_na_backup_entropy {0, 1}` (default: `0`)

Whether the SAC entropy bonus appears in the **action critic's** Bellman
target.

- `0` — `Q_a` target is `r + γ * Q_a_target(s', pi0(s', z'))` with no
  entropy term. Matches the reference jaxrl2 DSRL-NA default.
- `1` — `Q_a` target is `r + γ * (Q_a_target(s', pi0(s', z')) − α * log π(z'|s'))`.
  Matches the SB3 fork.

The actor's loss always includes the entropy term in either case; this flag
only affects the action critic target.

## Pi0 / openpi optimization

### `--save_kv_cache {0, 1}` (default: `0`)

Cache pi0's prompt-encoder K/V tensors per replay-buffer slot at rollout time.
At update time the cached K/V is reused so pi0's prompt encoder is **not**
re-run on the batch — only the diffusion stack runs. Cuts per-update pi0 cost
substantially on long prompts.

- `0` — re-encode pi0 prompts every update (slower, no extra memory).
- `1` — store K/V `(num_layers, num_tokens, ..., dim)` per slot via
  `agent_dp.get_prefix_rep_and_kv_cache(...)`. Requires the arayabrain
  openpi fork (the upstream openpi doesn't expose the K/V hook).

When set to `1`, the round-robin buffer's `CacheStorage` holds the K/V; the
`_na_update_step` injects `original_k_cache`/`original_v_cache` (current obs)
and `original_next_k_cache`/`original_next_v_cache` (next obs) into the batch
before the pi0 call.

## Distillation knobs

### `--grl_noise_sample {0, 1}` (default: `0`)

How fresh distillation noise `z_distill` is drawn each update.

- `0` — `z_distill ~ N(0, I)` always.
- `1` — 50/50 mixture of `N(0, I)` and `actor.sample(s)`. Drawn this way
  because once the actor has converged into a tight distribution, fresh
  Gaussian samples don't cover the latents the actor actually visits, so
  distilling on Gaussians alone yields a critic that is accurate on
  unreachable latents.

## Action-magnitude wiring

### `--noise_scale_inside {0, 1}` (default: `1`)

Where the action_magnitude bound is enforced relative to the actor's
squashed Gaussian.

- `1` — bound the actor distribution to
  `[-action_magnitude, +action_magnitude]` internally; the actor's
  `tanh * action_magnitude` returns the final action, log-prob accounts
  for the scale. Matches our existing DSRL-SAC.
- `0` — bound the actor to `[-1, 1]`, then multiply by `action_magnitude`
  externally (the multiplier never appears inside the policy distribution).
  Matches the arayabrain reference default. This affects the entropy term
  via the change-of-variables Jacobian, so toggling it is **not** a no-op
  for SAC.

`--action_magnitude` itself comes from `train_args_dict` (default `1.0`).

## Per-component update toggles (ablations)

These three flags let you turn each piece of the NA update off independently
to isolate which component is contributing.

### `--train_action_critic {0, 1}` (default: `1`)
If `0`, the action critic loss is skipped. The Bellman target for `Q_a` is
not computed and pi0 is **not** run on `next_observations` for the action
critic that step.

### `--train_noise_actor {0, 1}` (default: `1`)
If `0`, the actor (noise policy) is not updated. Useful for distillation-only
sanity checks.

### `--train_noise_critic {0, 1}` (default: `1`)
If `0`, the noise critic distillation step is skipped. With this off, the
actor uses whatever `Q_n` it had at init (or last update). Mostly useful for
verifying the distillation loss is what's actually moving the actor.

## Action-space lift (algorithm-agnostic but tested with NA)

### `--sac_action_chunk_size INT` (default: `1`)

Number of 32-d noise latents the SAC actor outputs per chunk. Decoupled from
`--query_freq` so the original (un-lifted) reference behaviour stays
invokable.

- `1` — reference DSRL/DSRL-NA shape `(1, 32)`. The single latent is
  broadcast across all 50 rows of pi0's diffusion noise tensor. This is
  what both the original `dsrl_pi0` and the arayabrain `dsrl_pi0_na`
  reference repos use; neither has a "lift" knob.
- `> 1` — lifted shape `(sac_action_chunk_size, 32)`. SAC controls one
  32-d latent per row; the trailing `50 - sac_action_chunk_size` rows of
  pi0's noise are filled by repeating the last SAC-controlled row. Set to
  `query_freq` to give every executed env-step its own SAC latent. SAC
  actor-head output grows linearly with this knob (`sac_action_chunk_size *
  32 * 2` for mean+log_std).

Range: `[1, 50]` (50 = pi0's `action_chunk_size`).

`--query_freq` is **not** changed by this knob: it still controls how many
env-actions are executed per pi0 chunk, the buffer's per-transition discount
(`γ ** query_freq`), and the diffused-action-critic chunk shape
`(query_freq, env_action_dim)`.

## Quick-reference table

| Flag | Default | Effect when `1` |
|------|---------|-----------------|
| `--algorithm pixel_dsrl_na` | `pixel_sac` | Switches to two-critic learner + NA buffer |
| `--sac_action_chunk_size` | `1` | `>1` lifts SAC noise output from `(1,32)` to `(N,32)` |
| `--noise_critic_grad_steps` | `1` | Reserved (inner distillation step count) |
| `--dsrl_na_backup_entropy` | `0` | Add SAC entropy bonus to `Q_a` Bellman target |
| `--save_kv_cache` | `0` | Cache pi0 K/V per slot (needs arayabrain openpi) |
| `--grl_noise_sample` | `0` | Mix actor-sampled noise into distillation noise |
| `--noise_scale_inside` | `1` | Bound actor dist internally vs. external multiply |
| `--train_action_critic` | `1` | Toggle action-critic update (ablation) |
| `--train_noise_actor` | `1` | Toggle actor update (ablation) |
| `--train_noise_critic` | `1` | Toggle distillation update (ablation) |

## Recommended starting points

- **Smoke / debug** (`examples/scripts/run_libero_robometer_na.sh`):
  ```
  --algorithm pixel_dsrl_na --save_kv_cache 0 --noise_scale_inside 1
  --grl_noise_sample 0 --noise_critic_grad_steps 1
  ```
  Closest to our DSRL-SAC behavior; isolates the two-critic structure from
  pi0 K/V cache complications.

- **Production** (matches arayabrain reference more closely):
  ```
  --algorithm pixel_dsrl_na --save_kv_cache 1 --noise_scale_inside 0
  --grl_noise_sample 1 --noise_critic_grad_steps 10 --hidden_dims 2048 2048 2048
  --action_magnitude 1.5
  ```
  Note `--noise_critic_grad_steps 10` is currently a no-op; flip it on once
  the inner distillation loop is wired up.
