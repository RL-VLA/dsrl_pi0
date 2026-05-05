"""Unit tests for the scattered training-samples pipeline.

No robometer server, no JAX, no libero env — these tests only exercise the
sample-construction logic via ``MockScatteredScorer`` + the helpers in
``jaxrl2.data.scattered_samples`` + the existing reward fns from
``examples.train_utils_sim_robometer``.

See ``docs/scattered_training_samples_plan.md`` for the design.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.scattered_robometer import (  # noqa: E402
    MockScatteredScorer,
    ScatteredScoreResult,
)
from jaxrl2.data import scattered_samples as ss  # noqa: E402
from examples.train_utils_sim_robometer import (  # noqa: E402
    _libero_sparse_batched,
    _libero_success_plus_robo_progress_batched,
    _progress_delta_batched,
    _progress_level_batched,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic per-env-step streams for a 40-step "trajectory"
# ---------------------------------------------------------------------------

def _make_streams(T: int = 40, noise_dim: int = 32, action_dim: int = 7):
    """Build synthetic obs/noise/executed streams + a libero-rewards array."""
    rng = np.random.default_rng(123)
    noise_stream = rng.normal(size=(T, noise_dim)).astype(np.float32)
    executed_stream = rng.normal(size=(T, action_dim)).astype(np.float32)
    obs_stream = [
        {
            "pixels": np.full((1, 64, 64, 3, 1), t, dtype=np.uint8),
            "state": np.full((1, 9, 1), t, dtype=np.float32),
        }
        for t in range(T)
    ]
    libero_rewards = np.zeros((T,), dtype=np.float32)
    libero_rewards[T - 1] = 1.0  # success at the last step
    return obs_stream, noise_stream, executed_stream, libero_rewards


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_interpolate_progress_anchors_zero_at_origin():
    sel = np.array([5, 10], dtype=np.int64)
    val = np.array([0.5, 1.0], dtype=np.float32)
    out = ss.interpolate_progress(sel, val, T=20)
    # Anchor at frame 0 should always be 0.
    assert out[0] == pytest.approx(0.0)
    # Linear between (0, 0) and (5, 0.5):
    assert out[2] == pytest.approx(0.2, abs=1e-5)
    assert out[5] == pytest.approx(0.5, abs=1e-5)
    # Linear between (5, 0.5) and (10, 1.0):
    assert out[7] == pytest.approx(0.7, abs=1e-5)
    # Right edge: held constant past last anchor.
    assert np.all(out[10:] == pytest.approx(1.0))


@pytest.mark.unit
def test_interpolate_progress_overrides_nonzero_at_origin():
    """Even if the scorer claims progress[0] > 0, we pin it to 0."""
    sel = np.array([0, 10], dtype=np.int64)
    val = np.array([0.3, 1.0], dtype=np.float32)
    out = ss.interpolate_progress(sel, val, T=11)
    assert out[0] == pytest.approx(0.0)
    assert out[10] == pytest.approx(1.0)


@pytest.mark.unit
def test_interpolate_progress_validates_inputs():
    with pytest.raises(ValueError):
        ss.interpolate_progress(np.array([5, 3]), np.array([0.1, 0.2]), T=10)
    with pytest.raises(ValueError):
        ss.interpolate_progress(np.array([5, 12]), np.array([0.1, 0.2]), T=10)


@pytest.mark.unit
def test_interpolate_success_step_semantics_never_decreases():
    sel = np.array([3, 6, 9], dtype=np.int64)
    val = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # spurious dip at t=9
    out = ss.interpolate_success(sel, val, T=12)
    assert np.all(np.diff(out) >= 0.0), "success curve must be non-decreasing"
    assert out[5] == pytest.approx(0.0)
    assert out[6] == pytest.approx(1.0)
    assert out[11] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MockScatteredScorer
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_mock_uniform_random_anchors_zero_and_is_sorted():
    scorer = MockScatteredScorer(
        num_selected=5, strategy="uniform_random",
        progress_oracle="monotone_random", emit_success=False, seed=7,
    )
    video = np.zeros((40, 1, 1, 3), dtype=np.uint8)
    res = scorer.score(video=video, task="t", sample_id="ep_0")
    assert isinstance(res, ScatteredScoreResult)
    assert res.selected_indices[0] == 0
    assert np.all(np.diff(res.selected_indices) > 0)
    assert res.selected_indices[-1] < 40
    assert res.success is None  # emit_success=False


@pytest.mark.unit
def test_mock_libero_reward_normalised_progress_is_monotone():
    T = 40
    libero_rewards = np.zeros((T,), dtype=np.float32)
    libero_rewards[15:] = 0.5  # reward starts mid-episode
    libero_rewards[T - 1] = 1.0
    scorer = MockScatteredScorer(
        num_selected=8, strategy="uniform_grid",
        progress_oracle="libero_reward_normalised", emit_success=True, seed=0,
    )
    video = np.zeros((T, 1, 1, 3), dtype=np.uint8)
    res = scorer.score(
        video=video, task="t", sample_id="ep_libero",
        libero_rewards=libero_rewards, libero_success_step=T - 1,
    )
    assert np.all(np.diff(res.progress) >= -1e-7), "progress must be non-decreasing"
    assert res.progress[0] == pytest.approx(0.0)
    assert res.success is not None
    assert res.success[-1] == pytest.approx(1.0)


@pytest.mark.unit
def test_mock_is_deterministic_under_fixed_seed():
    scorer = MockScatteredScorer(
        num_selected=6, seed=42, emit_success=False,
        progress_oracle="monotone_random",
    )
    video = np.zeros((30, 1, 1, 3), dtype=np.uint8)
    a = scorer.score(video=video, task="t", sample_id="ep_3", libero_rewards=None)
    b = scorer.score(video=video, task="t", sample_id="ep_3", libero_rewards=None)
    np.testing.assert_array_equal(a.selected_indices, b.selected_indices)
    np.testing.assert_array_equal(a.progress, b.progress)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_construct_look_future_basic():
    obs, noise, ex, _ = _make_streams(T=40)
    selected = np.array([0, 10, 20, 30], dtype=np.int64)
    progress_at = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
    progress_dense = ss.interpolate_progress(selected, progress_at, T=40)
    success_dense = np.zeros(40, dtype=np.float32)
    H = 5
    samples = ss.construct_look_future(
        selected_indices=selected,
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=H, env="libero",
    )
    # 4 candidates, all f < T-1=39, so all 4 survive.
    assert len(samples) == 4
    s0 = samples[0]
    assert s0.start == 0 and s0.end == 5
    assert not s0.padded
    np.testing.assert_array_equal(s0.noise, noise[0:5])
    np.testing.assert_array_equal(s0.executed, ex[0:5])
    assert s0.progress_pre == pytest.approx(progress_dense[0])
    assert s0.progress_post == pytest.approx(progress_dense[5])


@pytest.mark.unit
def test_construct_look_future_right_edge_padding():
    """A selected index near the end forces right-pad of noise+executed."""
    obs, noise, ex, _ = _make_streams(T=20)
    selected = np.array([0, 18], dtype=np.int64)
    progress_dense = ss.interpolate_progress(selected, np.array([0.0, 0.9], dtype=np.float32), T=20)
    success_dense = np.zeros(20, dtype=np.float32)
    H = 5
    samples = ss.construct_look_future(
        selected_indices=selected,
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=H, env="libero",
    )
    assert len(samples) == 2
    edge = samples[-1]
    assert edge.start == 18 and edge.end == 23
    assert edge.padded is True
    assert edge.noise.shape == (H, noise.shape[1])
    assert edge.executed.shape == (H, ex.shape[1])
    # Libero noise pads with repeat-last; executed pads with zeros.
    np.testing.assert_array_equal(edge.executed[2:], np.zeros((3, ex.shape[1]), dtype=np.float32))
    np.testing.assert_array_equal(edge.noise[2:], np.tile(noise[19], (3, 1)))


@pytest.mark.unit
def test_construct_look_future_drops_terminal_index():
    obs, noise, ex, _ = _make_streams(T=20)
    selected = np.array([0, 19], dtype=np.int64)  # 19 == T-1 → dropped
    progress_dense = ss.interpolate_progress(selected, np.array([0.0, 1.0], dtype=np.float32), T=20)
    success_dense = np.zeros(20, dtype=np.float32)
    samples = ss.construct_look_future(
        selected_indices=selected,
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=5, env="libero",
    )
    assert len(samples) == 1


@pytest.mark.unit
def test_construct_look_history_left_edge_padding():
    obs, noise, ex, _ = _make_streams(T=20)
    selected = np.array([0, 3, 12], dtype=np.int64)  # f=0 dropped; f=3 left-pads
    progress_dense = ss.interpolate_progress(selected, np.array([0.0, 0.1, 0.6], dtype=np.float32), T=20)
    success_dense = np.zeros(20, dtype=np.float32)
    H = 5
    samples = ss.construct_look_history(
        selected_indices=selected,
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=H, env="libero",
    )
    # f=0 dropped, f=3 and f=12 survive.
    assert len(samples) == 2
    left = samples[0]
    assert left.start == -2 and left.end == 3
    assert left.padded is True
    # Left-pad: noise repeats noise[0]; executed zeros for the off-edge entries.
    np.testing.assert_array_equal(left.noise[:2], np.tile(noise[0], (2, 1)))
    np.testing.assert_array_equal(left.executed[:2], np.zeros((2, ex.shape[1]), dtype=np.float32))
    np.testing.assert_array_equal(left.noise[2:], noise[0:3])
    np.testing.assert_array_equal(left.executed[2:], ex[0:3])


@pytest.mark.unit
def test_construct_random_subsample_in_range_only():
    obs, noise, ex, _ = _make_streams(T=40)
    progress_dense = np.linspace(0.0, 1.0, 40).astype(np.float32)
    success_dense = np.zeros(40, dtype=np.float32)
    H = 5
    rng = np.random.default_rng(0)
    samples = ss.construct_random_subsample(
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=H, env="libero", n=10, rng=rng,
    )
    assert len(samples) == 10
    for s in samples:
        assert 0 <= s.start <= 40 - H
        assert s.end == s.start + H
        assert not s.padded
        assert s.progress_pre == pytest.approx(progress_dense[s.start])
        assert s.progress_post == pytest.approx(progress_dense[s.end])


# ---------------------------------------------------------------------------
# Reward-fn reuse via the synthetic ScoreResult
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_apply_reward_fn_progress_delta_matches_per_sample_post_minus_pre():
    obs, noise, ex, _ = _make_streams(T=40)
    selected = np.array([0, 10, 20, 30], dtype=np.int64)
    progress_at = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
    progress_dense = ss.interpolate_progress(selected, progress_at, T=40)
    success_dense = np.zeros(40, dtype=np.float32)
    H = 5
    samples = ss.construct_look_future(
        selected_indices=selected,
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=H, env="libero",
    )
    rewards, masks = ss.apply_reward_fn_scattered(
        _progress_delta_batched, samples,
        libero_is_success=False, libero_success_step=None,
        H=H, robometer_success_threshold=0.5, emit_success=False,
        reward_kind="progress_delta",
    )
    expected = np.array(
        [s.progress_post - s.progress_pre for s in samples], dtype=np.float32
    )
    # Per-sample T=2 trick: every sample's reward is progress_post - progress_pre.
    # No terminal handling because libero_is_success=False.
    np.testing.assert_allclose(rewards, expected, atol=1e-6)
    assert np.all(masks == 1.0)


@pytest.mark.unit
def test_apply_reward_fn_progress_level_post_chunk_credit():
    obs, noise, ex, _ = _make_streams(T=40)
    selected = np.array([0, 10, 20, 30], dtype=np.int64)
    progress_at = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
    progress_dense = ss.interpolate_progress(selected, progress_at, T=40)
    success_dense = np.zeros(40, dtype=np.float32)
    H = 5
    samples = ss.construct_look_future(
        selected_indices=selected,
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=H, env="libero",
    )
    rewards, _ = ss.apply_reward_fn_scattered(
        _progress_level_batched, samples,
        libero_is_success=False, libero_success_step=None,
        H=H, robometer_success_threshold=0.5, emit_success=False,
        reward_kind="progress",
    )
    # Per-sample T=2 with post-shift: reward[k] = progress_post[k] exactly.
    expected_post = np.array([s.progress_post for s in samples], dtype=np.float32)
    np.testing.assert_allclose(rewards, expected_post, atol=1e-6)


@pytest.mark.unit
def test_apply_reward_fn_libero_success_terminal_mask_redirects_to_straddling_sample():
    obs, noise, ex, _ = _make_streams(T=40)
    selected = np.array([0, 10, 20, 30], dtype=np.int64)
    progress_at = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float32)
    progress_dense = ss.interpolate_progress(selected, progress_at, T=40)
    success_dense = np.zeros(40, dtype=np.float32)
    H = 5
    samples = ss.construct_look_future(
        selected_indices=selected,
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=H, env="libero",
    )
    # Success straddles sample at index 2 (start=20, end=25).
    success_step = 22
    rewards, masks = ss.apply_reward_fn_scattered(
        _libero_success_plus_robo_progress_batched, samples,
        libero_is_success=True, libero_success_step=success_step,
        H=H, robometer_success_threshold=0.5, emit_success=False,
        reward_kind="libero_success_plus_robo_progress",
    )
    assert masks[2] == pytest.approx(0.0)
    others = np.delete(masks, 2)
    assert np.all(others == 1.0)
    # Straddling sample gets the sparse base flipped from -1 to 0; non-
    # straddling samples keep -1 + progress_post.
    expected_straddling = 0.0 + samples[2].progress_post
    expected_others = np.array(
        [-1.0 + samples[k].progress_post for k in range(len(samples)) if k != 2],
        dtype=np.float32,
    )
    assert rewards[2] == pytest.approx(expected_straddling, abs=1e-6)
    np.testing.assert_allclose(np.delete(rewards, 2), expected_others, atol=1e-6)


@pytest.mark.unit
def test_apply_reward_fn_libero_sparse_terminal_only_at_straddling_sample():
    obs, noise, ex, _ = _make_streams(T=40)
    selected = np.array([0, 10, 20, 30], dtype=np.int64)
    progress_dense = ss.interpolate_progress(
        selected, np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), T=40
    )
    success_dense = np.zeros(40, dtype=np.float32)
    H = 5
    samples = ss.construct_look_future(
        selected_indices=selected,
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=H, env="libero",
    )
    success_step = 11  # straddles sample with start=10, end=15 → index 1
    rewards, masks = ss.apply_reward_fn_scattered(
        _libero_sparse_batched, samples,
        libero_is_success=True, libero_success_step=success_step,
        H=H, robometer_success_threshold=0.5, emit_success=False,
        reward_kind="libero_sparse",
    )
    assert masks[1] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.0)
    assert np.all(np.delete(masks, 1) == 1.0)
    assert np.all(np.delete(rewards, 1) == -1.0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_random_subsample_is_deterministic_under_fixed_rng():
    obs, noise, ex, _ = _make_streams(T=40)
    progress_dense = np.linspace(0.0, 1.0, 40).astype(np.float32)
    success_dense = np.zeros(40, dtype=np.float32)
    a = ss.construct_random_subsample(
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=5, env="libero", n=8, rng=np.random.default_rng(1234),
    )
    b = ss.construct_random_subsample(
        obs_stream=obs, noise_stream=noise, executed_stream=ex,
        progress_dense=progress_dense, success_dense=success_dense,
        H=5, env="libero", n=8, rng=np.random.default_rng(1234),
    )
    assert [s.start for s in a] == [s.start for s in b]
