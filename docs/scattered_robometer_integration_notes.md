# Frame-Aware Robometer — Phase 2 Integration Notes

Context: `docs/scattered_training_samples_plan.md` shipped Phase 1 — full
client-side pipeline + `MockScatteredScorer` + tests. Phase 2 replaces the mock
with a real frame-aware ("scattered") robometer endpoint and wires the existing
`RobometerClient` to call it.

This doc captures the integration plan, draft request/response wire format, and
draft Python API on both sides. Goal: a frame-aware endpoint can be merged into
robometer + this client without rethinking the contract.

---

## 1. What the client already expects

The client-side scattered pipeline is locked in. Phase 2 only needs to plug a
real scorer into the existing seam.

### 1.a Seam in this repo

`examples/train_utils_sim_robometer.py::_score_and_insert_scattered` constructs a
`MockScatteredScorer` and calls:

```python
score = scorer.score(
    video=video_stub,                  # (T, H, W, C) uint8 — full episode
    task=str(variant.task_description),
    sample_id=f"ep_{...}",
    libero_rewards=...,                # mock-only oracle, drop in real client
    libero_success_step=...,           # mock-only oracle, drop in real client
)
```

It then consumes `score.selected_indices`, `score.progress`, `score.success`.

The Protocol that defines this seam is already in
`examples/scattered_robometer.py::ScatteredScorer`. The real client will simply
implement it.

### 1.b The dataclass the rest of the codebase depends on

```python
@dataclass(frozen=True)
class ScatteredScoreResult:
    id: str
    selected_indices: np.ndarray   # (K,) int64, sorted strictly increasing, in [0, T)
    progress: np.ndarray           # (K,) float32 in [0, 1]
    success: Optional[np.ndarray]  # (K,) float32 or None
```

`jaxrl2/data/scattered_samples.py::interpolate_progress / interpolate_success`
both pin frame 0 to progress 0 (per task spec) regardless of what the server
emits, so the server is free to omit frame 0 from `selected_indices`.

---

## 2. Phase 2 deliverables

| Side | Deliverable | Owner |
|---|---|---|
| Server | New endpoint `POST /score_scattered` (or flag on `/score`) | robometer |
| Server | Decision: how `K` is chosen, whether it's adaptive or capped | robometer |
| Server | Backwards-compat: existing `/score` keeps the chunk-grid contract | robometer |
| Client (pkg) | New method `RobometerClient.score_scattered(...)` returning canonical `ScatteredScoreResult` | robometer_client (vendored at `third_party/robometer_client/`) |
| Client (pkg) | Move `ScatteredScoreResult` into `robometer_client.types` so it's the canonical type | robometer_client |
| Client (pkg) | Async variant `submit_scattered(...)` mirroring `submit(...)` | robometer_client |
| dsrl_pi0 | Replace `MockScatteredScorer` with `RobometerClient.score_scattered` in `_score_and_insert_scattered` | this repo |
| dsrl_pi0 | Wire async path: `PendingScatteredScores` mirroring `PendingScores` | this repo |
| dsrl_pi0 | Drop the local `examples/scattered_robometer.ScatteredScoreResult` and import from `robometer_client.types` | this repo |
| dsrl_pi0 | Keep `MockScatteredScorer` in tests only (`tests/test_scattered_samples.py`) | this repo |

---

## 3. Draft wire protocol

The existing `/score` endpoint accepts a multipart upload with `(VideoSample,
use_frame_steps)` and returns `ScoreResult`s. Frame-aware adds a separate
endpoint to keep the existing contract clean.

### 3.a `POST /score_scattered`

**Request (multipart, mirrors `/score`):**

| Part | Type | Notes |
|---|---|---|
| `video_<i>.mp4` (or `.npz`) | binary | One per `VideoSample`. Same encoding as `/score`. |
| `meta` | JSON | Per-batch metadata, same as `/score`. |
| `params` | JSON | Frame-aware specific params (see below). |

**`params` JSON (request-level, applies to all videos in the batch):**

```json
{
  "max_selected": 32,
  "min_selected": 4,
  "selection_strategy": "auto",
  "emit_success": true,
  "include_endpoints": false
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `max_selected` | int | 32 | Upper bound on `K`. Server may return fewer. |
| `min_selected` | int | 2 | Lower bound on `K`. Server pads with uniform-grid picks if its native saliency is sparser. |
| `selection_strategy` | str | `"auto"` | `"auto"` lets the server choose; alternatives are server-defined (`"saliency"`, `"uniform_grid"`, `"saliency_plus_endpoints"`, ...). |
| `emit_success` | bool | true | If false, omit `success` arrays from the response. |
| `include_endpoints` | bool | false | If true, server is REQUIRED to include indices `0` and `T-1`. If false, client pins frame 0 to progress 0 itself; right-edge held constant. |

**Response (JSON, list aligned to request video order):**

```json
{
  "results": [
    {
      "id": "ep_42",
      "T": 400,
      "selected_indices": [0, 23, 47, 89, 134, ...],
      "progress":         [0.0, 0.05, 0.18, 0.41, 0.62, ...],
      "success":          [0.0, 0.0, 0.0, 0.0, 0.1, ...]
    },
    ...
  ],
  "model_id": "robometer-libero-v3",
  "version": "0.4.1"
}
```

**Per-result invariants the client relies on:**

1. `len(selected_indices) == len(progress)`.
2. `len(success) == len(selected_indices)` when present, otherwise `success` is omitted (key absent or `null`).
3. `selected_indices` is **sorted strictly increasing** with all values in `[0, T)`.
4. `progress[k] in [0.0, 1.0]`.
5. `success[k] in [0.0, 1.0]` (probability, not a hard bool).
6. `T` matches the number of frames the server actually consumed (post any
   internal subsampling) — purely informational, not used for indexing.

The client validates (1)–(5) on receipt and raises `ValueError` if violated.
Invariant (6) is logged but not enforced.

### 3.b Errors

Same shape as `/score`: HTTP 4xx/5xx with `{"error": "...", "code": "..."}`.
Existing `RobometerClient` retry logic applies unchanged.

### 3.c Why a separate endpoint instead of a flag

- Different output shape (`progress[T]` vs `progress[K]` + `selected_indices[K]`).
- Different server compute path (saliency selector → progress head).
- `use_frame_steps` on the existing `/score` is a model-side knob, not a
  selector-side one — overloading it would conflate two orthogonal axes.
- Keeps the existing `/score` byte-for-byte stable so chunk-grid runs are
  unchanged when the new endpoint deploys.

A flag on `/score` is acceptable as an alternative if the server team prefers,
but the client API will still expose two methods (`score` and
`score_scattered`) returning two different types.

---

## 4. Draft Python API on the client side

### 4.a New canonical type in `robometer_client.types`

```python
@dataclass(frozen=True)
class ScatteredScoreResult:
    """Per-video result from the frame-aware ('scattered') endpoint.

    selected_indices: (K,) int64 env-step indices into the submitted video,
        sorted strictly increasing, all in [0, T).
    progress: (K,) float32 in [0, 1] at the selected indices.
    success: (K,) float32 in [0, 1] at the selected indices, or None when the
        loaded model has no success head or emit_success=False was requested.
    T: total number of frames in the submitted video. Informational.
    model_id, version: server identity for reproducibility.
    """
    id: str
    selected_indices: np.ndarray
    progress: np.ndarray
    success: Optional[np.ndarray]
    T: int
    model_id: str
    version: str
```

Note: this replaces `examples.scattered_robometer.ScatteredScoreResult`. The
fields `T`, `model_id`, `version` are added for forward-compat. The Phase-1
client code only reads `id`, `selected_indices`, `progress`, `success`, so the
extra fields are no-ops in our consumers.

### 4.b `RobometerClient` extensions

```python
class RobometerClient:
    # ... existing methods unchanged ...

    def score_scattered(
        self,
        samples: List[VideoSample],
        *,
        max_selected: int = 32,
        min_selected: int = 2,
        selection_strategy: str = "auto",
        emit_success: bool = True,
        include_endpoints: bool = False,
    ) -> List[ScatteredScoreResult]:
        """Synchronous frame-aware scoring. Mirrors `score(...)` shape."""

    def submit_scattered(
        self,
        samples: List[VideoSample],
        *,
        max_selected: int = 32,
        min_selected: int = 2,
        selection_strategy: str = "auto",
        emit_success: bool = True,
        include_endpoints: bool = False,
    ) -> Future[List[ScatteredScoreResult]]:
        """Async fire-and-forget. Mirrors `submit(...)` so `PendingScatteredScores`
        can reuse the same Future plumbing as `PendingScores`."""
```

### 4.c Adapter for the `ScatteredScorer` Protocol

The Phase-1 Protocol takes per-video input; `RobometerClient` is per-batch. A
small adapter keeps `_score_and_insert_scattered` unchanged:

```python
class RobometerScatteredScorer:
    """Adapter that satisfies examples.scattered_robometer.ScatteredScorer
    using a real RobometerClient."""

    def __init__(
        self,
        client: RobometerClient,
        *,
        max_selected: int = 32,
        min_selected: int = 2,
        selection_strategy: str = "auto",
        emit_success: bool = True,
        include_endpoints: bool = False,
    ) -> None:
        self.client = client
        self.kwargs = dict(
            max_selected=max_selected,
            min_selected=min_selected,
            selection_strategy=selection_strategy,
            emit_success=emit_success,
            include_endpoints=include_endpoints,
        )

    def score(
        self,
        video: np.ndarray,
        task: str,
        sample_id: str,
        *,
        libero_rewards=None,        # ignored — real server doesn't need oracles
        libero_success_step=None,   # ignored
    ) -> ScatteredScoreResult:
        sample = VideoSample(frames=video, task=task, id=sample_id)
        [result] = self.client.score_scattered([sample], **self.kwargs)
        return result
```

This is the seam where `MockScatteredScorer` gets swapped out. The
`libero_rewards` / `libero_success_step` kwargs become no-ops at the real
adapter; tests still pass them to the mock.

---

## 5. Async batching pipeline (`PendingScatteredScores`)

Phase 1 runs the mock **synchronously** because it's in-process. Phase 2 must
overlap server scoring with the next rollout, mirroring the existing
`PendingScores` queue.

Sketch (mirrors `PendingScores` 1-for-1):

```python
class PendingScatteredScores:
    def __init__(self, client: RobometerClient, *, kwargs, fail_behavior="raise"):
        ...

    def submit_episode(self, traj: Dict, task: str, sample_id: str) -> None:
        sample = VideoSample(frames=_resize(traj["full_video"], ...), task=task, id=sample_id)
        future = self.client.submit_scattered([sample], **self.kwargs)
        self._pending.append((traj, future))

    def drain_ready(self) -> List[Tuple[Dict, ScatteredScoreResult]]: ...
    def drain_all(self) -> List[Tuple[Dict, ScatteredScoreResult]]: ...
    def popleft_oldest(self) -> Optional[Tuple[Dict, ScatteredScoreResult]]: ...
```

The training-loop branch becomes:

```python
if is_scattered:
    pending_scattered.submit_episode(traj, task=..., sample_id=...)
    while len(pending_scattered) > variant.scattered_queue_max_depth:
        traj_done, score = pending_scattered.popleft_oldest()
        total_env_steps += _insert_scattered(variant, traj_done, score, ...)
    for traj_done, score in pending_scattered.drain_ready():
        total_env_steps += _insert_scattered(variant, traj_done, score, ...)
```

`_insert_scattered` is the existing scoring logic in
`_score_and_insert_scattered`, refactored to take a pre-computed `score` instead
of building one inline. The mock path can keep the in-process
`MockScatteredScorer` route for offline testing.

---

## 6. Required vs. optional fields — a strict contract

Hard requirements on the server response (client raises if violated):

| Field | Required | Notes |
|---|---|---|
| `id` | yes | Echoes the input `VideoSample.id` (or positional index). |
| `selected_indices` | yes | Strictly increasing int array, all in `[0, T)`. |
| `progress` | yes | Same length as `selected_indices`, values in `[0, 1]`. |
| `success` | iff `emit_success=true` AND model has success head | Same length and bounds as `progress`. |
| `T` | yes | Plain int. |
| `model_id` | yes | Free-form string. |
| `version` | yes | Free-form string. |

Optional fields (server may omit; client must tolerate absence):

- `selection_meta`: free-form server diagnostics (e.g. saliency scores per
  selected index). Logged but not consumed.
- `T_consumed`: if the server downsampled the video before scoring, this is
  the post-downsample frame count. Logged but not used for indexing.

---

## 7. Client-side validation checklist

Add to `RobometerClient.score_scattered` immediately after JSON parse:

1. Top-level: `results` is a list whose length equals the input batch size.
2. Per result:
   - `selected_indices` is non-empty.
   - `selected_indices` is strictly increasing (`np.all(np.diff(idx) > 0)`).
   - `selected_indices.min() >= 0` and `selected_indices.max() < T`.
   - `len(progress) == len(selected_indices)`.
   - `progress.min() >= 0.0 - eps` and `progress.max() <= 1.0 + eps` (eps for
     server float jitter; clip on read).
   - If `emit_success=true` was requested AND the response includes `success`:
     same shape as `progress`, same bounds.
3. On any violation: raise `ValueError` carrying the offending field and
   `sample_id`. The training loop's `fail_behavior` (`raise` vs `drop`)
   decides what happens next, just like for `/score`.

---

## 8. Migration plan in this repo

When the real endpoint is live:

1. **`third_party/robometer_client/`** — bump version, add
   `ScatteredScoreResult` + `score_scattered` + `submit_scattered`. Land tests
   for round-trip JSON.
2. **`examples/scattered_robometer.py`** — keep `MockScatteredScorer` for
   tests; remove the local `ScatteredScoreResult` dataclass and re-export from
   `robometer_client.types` so consumers transparently switch.
3. **`examples/train_utils_sim_robometer.py`** — replace
   `_score_and_insert_scattered` body's `MockScatteredScorer` instantiation
   with a `RobometerScatteredScorer` adapter when
   `variant.scattered_use_mock == 0`.
4. **Async** — add `PendingScatteredScores` (alongside `PendingScores`) and a
   new `--scattered_queue_max_depth` CLI knob. Default to 4 like the chunk-grid
   pendings queue.
5. **Tests** — keep `tests/test_scattered_samples.py` exercising the
   in-process mock. Add a thin integration test that spins the
   `RobometerScatteredScorer` adapter against `MockScatteredScorer` injected
   as the client (duck-typing via the Protocol).

---

## 9. Open questions for the robometer team

These don't block client-side work but need answers before the wire format
gets finalized:

1. **Is `K` capped server-side?** The client passes `max_selected` but if the
   server's saliency selector has a hard ceiling (e.g. 64 frames), document it
   and let the client validate.
2. **Is `progress` monotone non-decreasing?** Phase 1 mock is monotone; the
   real saliency selector may not be. Reward fns tolerate non-monotone
   (delta can go negative), but the success-detection threshold logic
   assumes the cumulative-progress-then-success interpretation.
3. **Does the server always include the last frame?** If yes, the
   `include_endpoints` flag is moot. If no, the client's right-edge held-
   constant interpolation is the right call (current behaviour).
4. **What happens if the model has no success head?** Today `/score` returns
   `success=None`. Mirror that: server omits the `success` key entirely.
5. **Per-sample vs per-batch params?** The draft puts `params` at request
   level. If different videos in a batch should use different
   `max_selected`, move `params` to per-video. Current Phase-1 callsite uses
   identical params per batch, so request-level is fine.
6. **Streaming?** Long videos may benefit from a chunked upload protocol.
   Out of scope for v1; revisit if libero rollouts grow past O(1k) frames.

---

## 10. TL;DR

- Client side already has the seam (`ScatteredScorer` Protocol +
  `ScatteredScoreResult` dataclass + `_score_and_insert_scattered`).
- Server side needs a new `POST /score_scattered` endpoint returning
  `selected_indices[K] + progress[K] (+ success[K])`.
- `robometer_client` package gets `score_scattered` / `submit_scattered`
  methods returning a canonical `ScatteredScoreResult` type.
- This repo swaps `MockScatteredScorer` for a `RobometerScatteredScorer`
  adapter, adds an async `PendingScatteredScores` queue, and the rest of the
  scattered pipeline (sample construction, reward fns, buffer insert) stays
  unchanged.
- Mock stays in `tests/` for offline determinism; `examples/scattered_robometer.py`
  can keep the mock as a development utility but should re-export the
  canonical type from `robometer_client.types`.
