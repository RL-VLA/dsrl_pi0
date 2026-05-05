# Scattered Training Samples — Implementation Plan

Source request: `tasks/scattered_training_samples.md`.

## 1. Goal

Today the buffer is built from a **uniform chunk grid**: every `query_freq` env steps
we capture one (state, noise-latent, frame) triple, and the robometer scores that
fixed grid. The new feature lets a future robometer pick `K` **uneven important
frames** out of the full `max_timesteps`-frame video and annotate progress only at
those frames; we then reconstruct SAC training samples aligned to those scattered
indices using one of three chunk-construction modes (look-future, look-history,
random-subsample on a dense-interpolated progress curve).

The task explicitly notes: there is **no real robometer endpoint yet** for scattered
output. We design a mock + the full client-side pipeline + tests now.

## 2. Locked design decisions

(Captured from the user via AskUserQuestion at plan time.)

| Decision | Choice |
|---|---|
| How modes combine within a run | **One mode at a time** — `scattered_mode ∈ {off, look_future, look_history, random_subsample}` |
| Supported SAC action shape | **Lifted only**: `sac_action_chunk_size == query_freq`. Fail loudly if scattered enabled with unlifted shape |
| Initial algorithm coverage | **DSRL-SAC (`pixel_sac`) only**. DSRL-NA deferred — its per-chunk pi0 K/V cache breaks under off-grid starts |
| Reward function semantics | **Reuse existing `robometer_reward_kind`** (`progress_delta`, `progress`, `libero_success_plus_robo_progress`, …). Scattered mode pre-computes the per-sample (progress_pre, progress_post, success_pre, success_post) from the dense-interpolated curve and packages them into a `ScoreResult`-shaped object so the existing reward fns work unchanged |

## 3. Off-by-one convention (carry-over)

Existing convention from `train_utils_sim_robometer.py`:

```
chunk_frames[k] is captured BEFORE chunk k executes (env-step k * query_freq).
server.progress[k] / .success[k] describe the state at the START of chunk k
   = OUTCOME of chunk k-1.
reward attached to action a_k uses progress_post[k] = progress[k+1].
```

Scattered mode preserves this: for a sample whose chunk runs over env-steps
`[start, start+query_freq)`, we deliver `progress_pre = interp[start]` and
`progress_post = interp[start + query_freq]` (with edge handling, see §6). Reward
fns then see the same `(progress_pre, progress_post)` semantics they already
expect.

## 4. Architecture overview

```
collect_traj (lifted, per-env-step storage)
        │
        ▼
ScatteredScorer (mock first; real later) ── selected_indices, progress_at_selected, success_at_selected
        │
        ▼
ProgressInterpolator (anchored at frame 0 → 0)
        │
        ▼
ScatteredSampleConstructor   ◄── scattered_mode
        │     ├─ look_future
        │     ├─ look_history
        │     └─ random_subsample
        ▼
"Synthetic chunks": list of (start_env_step, obs_at_start, noise_chunk, executed_actions, progress_pre, progress_post, success_pre, success_post, libero_step_at_start)
        │
        ▼
_build_synthetic_score_result  →  ScoreResult-shaped object whose progress[k] is progress_pre[k] and length matches synthetic chunk count
        │
        ▼
existing reward_fn (progress_delta, progress, libero_success_plus_robo_progress, …)
        │
        ▼
add_online_data_to_buffer (extended to accept synthetic chunks instead of original chunk grid)
```

Existing chunk-grid path remains the default (`scattered_mode='off'`).

## 5. Storage changes in `collect_traj`

Currently stored per-chunk: `obs_dict`, `actions_noise` (one 32-d latent or
`(query_freq, 32)` lifted), `diffused_actions[:query_freq]`, `chunk_frames[k]`.
For scattered we need per-env-step granularity.

Changes — **only when `variant.scattered_mode != 'off'`** (zero overhead in default
path):

1. **Per-env-step frame stream** — append `agentview_image[::-1, ::-1]` every step
   (`max_timesteps` frames, ~50 MB at 256² for libero — same scale as existing
   `image_list` already kept for visualisation, just at full resolution).
2. **Per-env-step obs_dict stream** — record `obs_to_img(obs, variant)` and
   optionally `obs_to_qpos(obs, variant)` every step so any sample start `f` has
   an obs to attach. (Currently we only keep one `obs_dict` per chunk.)
3. **Per-env-step noise stream** — in lifted mode each chunk already produces
   `actions_noise: (query_freq, 32)`. Concatenate across chunks → one
   `(max_timesteps, 32)` stream. Indexing this at any `[f:f+query_freq]` gives a
   well-defined chunk action. **This is why the lifted-only restriction matters.**
4. **Per-env-step executed-action stream** — `actions[t % query_freq]` at every
   step → `(max_timesteps, env_action_dim)`.
5. Keep recording `chunk_frames`/`chunk_env_steps` so the chunk-grid path is
   untouched.

Memory budget: a 400-step libero rollout adds ≈ 400×256×256×3 = 79 MB frames,
plus 400×32×4 = 50 KB latents, plus 400×7×4 = 11 KB env actions. Tractable.

## 6. ScatteredScorer — mock implementation

Live behind a Protocol so the real robometer endpoint can drop in later:

```python
@dataclass
class ScatteredScoreResult:
    id: str
    selected_indices: np.ndarray   # (K,) int64, strictly increasing, [0, T)
    progress: np.ndarray           # (K,) float32 in [0, 1] at the selected indices
    success: np.ndarray | None     # (K,) float32 or None

class ScatteredScorer(Protocol):
    def score(self, video: np.ndarray, task: str, sample_id: str) -> ScatteredScoreResult: ...
```

Mock — `MockScatteredScorer` — picks `K` indices from the full video using a
configurable strategy:

- `strategy='uniform_random'`: pick `K` random indices from `[1, T-1]`,
  sort, prepend 0. (Frame 0 is always pinned to progress 0 per the task spec.)
- `strategy='uniform_grid'`: equidistant indices — useful as a sanity check
  reproducing the existing chunk-grid behaviour when `K = T // query_freq`.
- `strategy='libero_milestones'`: for libero, derive selected indices from the
  per-step libero reward signal (jumps in reward → milestones). Useful for
  realistic-shaped tests.

Progress at selected indices, in the mock, is computed by the chosen
`progress_oracle`:

- `progress_oracle='libero_reward_normalised'`: cumulative-libero-reward / max,
  evaluated at the selected indices. Deterministic, mirrors what a "perfect"
  robometer would emit on libero data.
- `progress_oracle='monotone_random'`: random sorted values in [0,1], with `0` at
  index 0 — useful for testing interpolation/edge cases.

Success at selected indices in the mock: `success[k] = 1` once
`libero_success_step <= selected_indices[k]`, else 0. Disable with
`emit_success=False` to test the no-success-head path.

## 7. ProgressInterpolator

Single function; **ALWAYS pins frame 0 to progress 0** (task spec):

```python
def interpolate_progress(
    selected_indices: np.ndarray,    # (K,) int, possibly excluding 0
    progress_at_selected: np.ndarray, # (K,) float32
    T: int,                          # full episode length (max_timesteps)
) -> np.ndarray:                     # (T,) float32, dense per-env-step
    # 1. Anchor: prepend (0, 0.0) if 0 not in selected_indices.
    # 2. Right-edge: if T-1 not in selected_indices, extend the last value
    #    constantly (we don't extrapolate beyond the last observation).
    # 3. Linear interpolation between consecutive anchors.
```

Same shape for `interpolate_success`, but with **step interpolation** instead of
linear (success is a binary event, smoothing it is misleading):
`success_dense[t] = max(success_at_selected[k] for k where selected_indices[k] <= t)`.

## 8. Sample construction modes

Common variables: `T = max_timesteps`, `H = query_freq`, `interp[t] = interpolate_progress(...)`,
`succ[t] = interpolate_success(...)`. `K = len(selected_indices)`.

### 8.a `look_future`

For each robometer-selected index `f` in `selected_indices`:

- `start = f`, `chunk = [f, f + H)` (length `H`).
- `obs_at_start = stored_obs[f]`.
- `noise = noise_stream[f : f + H]`.
- `executed = executed_stream[f : f + H]`.
- `progress_pre = interp[f] = progress_at_selected[k]` (annotated, exact).
- `progress_post = interp[min(f + H, T - 1)]` (interpolated; if `f + H >= T` we
   reuse `interp[T-1]`).
- `success_pre = succ[f]`, `success_post = succ[min(f + H, T - 1)]`.
- **Edge padding** — when `f + H > T`:
  - libero uses delta/relative actions, so pad with zeros for the "executed"
    stream and **repeat the last** stored noise latent.
  - Length stays exactly `H` so the buffer's action-shape invariant holds.
  - We mark this sample with `padded=True` for diagnostics; reward stays as
    defined above.

Discard a candidate if `f >= T - 1` (no chunk to look forward over).

### 8.b `look_history`

For each `f`:

- `start = max(f - H, 0)`, `chunk = [start, f)`. If `f - H < 0` we pad on the
   left (zero-actions for relative; repeat-first for absolute) so the chunk
   length is still `H`.
- `obs_at_start = stored_obs[start]`.
- `noise = noise_stream[start:f]` (left-padded as above).
- `executed = executed_stream[start:f]` (left-padded).
- `progress_pre = interp[start]`, `progress_post = interp[f]` (annotated, exact).
- `success_pre = succ[start]`, `success_post = succ[f]`.

Discard if `f == 0` (the only annotated frame might be 0, in which case there's
no history to look back at).

### 8.c `random_subsample`

Independent of `selected_indices` for sample placement (uses dense interpolated
progress instead). Config: `scattered_random_n`.

```python
candidate_starts = rng.choice(np.arange(0, T - H + 1), size=N, replace=False)
```

For each `start`:

- `chunk = [start, start + H)`.
- `obs_at_start = stored_obs[start]`.
- `noise = noise_stream[start:start + H]`.
- `executed = executed_stream[start:start + H]`.
- `progress_pre = interp[start]`, `progress_post = interp[start + H]`.
- `success_pre = succ[start]`, `success_post = succ[start + H]`.

No edge handling required (we restrict starts to `[0, T - H]`).

### 8.d Edge-padding policy

Auto-derived from `variant.env`:

- libero, aloha → **delta/relative actions** → pad with zeros for executed
  stream and repeat-last for noise stream.
- (future absolute-action env) → pad with the last value for both.

Encode this as a tiny dispatch so adding an env later is a one-line change.

## 9. Synthetic ScoreResult and reward-fn reuse

For each constructed sample we have `(progress_pre, progress_post, success_pre,
success_post)`. The existing reward fns expect a `ScoreResult`-shaped object
whose `progress[k]` is the **state BEFORE chunk k**, with the off-by-one shift
applied internally via `_post_chunk_shift`.

To re-use them unchanged, we build a synthetic `ScoreResult` of length `K + 1`
where:

```
progress = [progress_pre[0], progress_pre[1], ..., progress_pre[K-1], progress_post[K-1]]
success  = analogous, or None when emit_success=False
```

…but each reward fn is currently called once per episode on `(B, T)` batches.
Cleanest path: **add a single new helper `_apply_reward_fn_scattered`** that, for
each scattered episode, builds the synthetic `ScoreResult` (with `T = K`) and
calls the same reward fn under the hood. The reward fn output `(K,)` rewards
attach 1-to-1 to the constructed samples.

This keeps `REWARD_FNS` untouched and makes scattered mode look like "samples
arrive on a different grid" rather than "different reward semantics".

Caveats handled in the helper:

- `libero_success_plus_robo_progress` and friends rely on
  `meta['libero_is_success']`. For scattered samples we keep the **episode-level**
  flag (an episode either succeeded or not) and only modify the terminal-mask
  rule: `mask = 0` only for samples whose chunk straddles `libero_success_step`.
- `progress_delta` already uses `progress[k+1] - progress[k]` so the synthetic
  `ScoreResult` length-`K+1` trick gives `K` deltas — exactly the per-sample
  reward we want.
- `_post_chunk_shift` applied to the synthetic array yields `progress_post`
  back, which matches what `progress` and `libero_success_plus_robo_progress`
  expect.

## 10. Buffer insertion for scattered samples

`add_online_data_to_buffer` currently iterates `range(episode_len)` and inserts
`(obs[t], action_noise[t], reward[t], next_obs[t+1], next_action_noise[t+1])`.
For scattered we replace the per-chunk indexing with the per-sample list:

```python
for k, sample in enumerate(scattered_samples):
    obs = sample.obs            # at sample.start
    next_obs = sample.next_obs  # at sample.start + H, with right-edge clamping
    insert(
        observations=obs,
        next_observations=next_obs,
        actions=sample.noise,                  # (H, 32)
        next_actions=next_sample.noise if k < K - 1 else sample.noise,
        rewards=rewards[k],
        masks=masks[k],
        discount=variant.discount ** H,
    )
```

The `next_actions` chaining is a question of taste — for scattered samples the
"next sample" isn't a Bellman-natural neighbour. We default to chaining samples
in the order they were constructed (matches current code's `actions[t+1]`
semantics) but flag this in the docstring as approximate.

## 11. Config knobs

Added to `examples/launch_train_sim_robometer.py`:

```
--scattered_mode          {off, look_future, look_history, random_subsample}   default off
--scattered_strategy      {uniform_random, uniform_grid, libero_milestones}    default uniform_random  (mock only)
--scattered_progress_oracle {libero_reward_normalised, monotone_random}        default libero_reward_normalised  (mock only)
--scattered_num_selected  K           default = T // query_freq
--scattered_random_n      N           default = T // query_freq                only for random_subsample
--scattered_emit_success  {0, 1}      default 1                                (mock only)
--scattered_seed          int         default = run seed                       (mock only)
--scattered_use_mock      {0, 1}      default 1                                until real endpoint exists
```

Validation:

- `scattered_mode != 'off'` → require `sac_action_chunk_size == query_freq`,
  else raise immediately at startup.
- `scattered_mode != 'off'` → require `algorithm == 'pixel_sac'` (reject
  `pixel_dsrl_na`) until DSRL-NA support is added.
- `scattered_mode != 'off'` and reward kind requires success → require
  `scattered_emit_success == 1`.

## 12. Files touched / added

Added:
- `jaxrl2/data/scattered_samples.py` — `ProgressInterpolator`, `interpolate_progress`,
  `interpolate_success`, `ScatteredSample` dataclass, the three constructors, and
  `_apply_reward_fn_scattered`.
- `examples/scattered_robometer.py` — `ScatteredScoreResult`, `ScatteredScorer`
  Protocol, `MockScatteredScorer` (uniform_random / uniform_grid /
  libero_milestones).
- `tests/test_scattered_samples.py` — pytest unit tests, see §13.

Modified:
- `examples/train_utils_sim_robometer.py` — add per-env-step recording arms in
  `collect_traj` (gated by `variant.scattered_mode != 'off'`), add scattered
  branch in the insert path. Existing chunk-grid behaviour unchanged.
- `examples/launch_train_sim_robometer.py` — new CLI args.
- `examples/scripts/run_libero_robometer.sh` — opt-in env vars for the new knobs
  (default scattered off).
- `docs/scattered_training_samples_plan.md` — this file.

## 13. Tests (pytest, with the mock — no robometer server required)

`tests/test_scattered_samples.py`:

1. **interpolate_progress** — anchors frame 0 → 0; linear between annotated
   indices; right-edge constant past the last annotation.
2. **interpolate_success** — step (max-so-far) semantics, never decreases.
3. **MockScatteredScorer.uniform_random** — selected indices strictly increasing,
   in `[0, T)`; first index is 0; honours `K`.
4. **MockScatteredScorer.libero_reward_normalised** — progress monotone
   non-decreasing; matches a hand-computed cumulative for a synthetic
   per-step-libero-reward array.
5. **look_future construction** — given fixed selected indices on a synthetic
   400-step trajectory, sample list has correct `start`, correct slice of
   `noise_stream`, correct `progress_pre = annotated`, correct
   `progress_post = interp[start + H]`. Edge sample where `start + H > T`
   exercises the padding branch (length still `H`, last noise repeated, executed
   zero-padded).
6. **look_history construction** — analogous; left-edge sample where
   `start = 0` and `f < H` exercises left padding.
7. **random_subsample** — `N` samples, all starts in `[0, T - H]`,
   `progress_pre/post = interp[start], interp[start + H]`.
8. **synthetic ScoreResult + progress_delta reward fn** — for a constructed
   sample list, `_apply_reward_fn_scattered(progress_delta, …)` yields rewards
   equal to `progress_post - progress_pre` per sample. End-to-end check that
   the reward fn reuse is correct.
9. **synthetic ScoreResult + libero_success_plus_robo_progress** — terminal mask
   only set on the sample whose chunk straddles `libero_success_step`.
10. **lifted-only guard** — calling the scattered constructor with
   `sac_action_chunk_size != query_freq` raises a clear error.
11. **DSRL-NA guard** — `algorithm='pixel_dsrl_na'` + scattered enabled raises a
   clear error at startup (config-validation level test).
12. **Determinism** — fixed seed → identical samples across runs.

## 14. Phasing

- **Phase 1** (this PR target): all of §5–§13. Default `scattered_mode='off'`,
  zero behavioural change for existing runs.
- **Phase 2** (separate PR): real robometer endpoint emitting
  `ScatteredScoreResult`. Drop `MockScatteredScorer` from the live code path,
  keep it for tests.
- **Phase 3** (separate PR): DSRL-NA support — re-extract pi0 K/V cache at
  scattered starts (the simplest correct option) or revisit the design for
  cached re-use.

## 15. Open questions parked for later

- Concrete number of robometer-selected frames per video (currently a knob —
  task spec mentions ~20/400). The real robometer will set this; the mock honours
  whatever `K` is configured.
- Exact strategy mix in the mock for production-shaped tests (libero milestones
  vs uniform random vs heuristic) — easy to extend.
- Whether to also pad the **observation** stream on right-edge `look_future`
  samples (we currently just clamp `next_obs` to `obs[T-1]`).
