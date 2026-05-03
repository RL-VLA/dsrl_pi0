"""Unit tests for the SAC action-shape lift from (1, 32) to (query_freq, 32).

These tests don't load pi0 weights or step a libero env — they exercise the
plumbing only (DummyEnv shape, PixelSACLearner derived attrs, ReplayBuffer
round-trip, and the noise-build pad-and-concatenate path used by collect_traj).

See docs/sac_action_lift_plan.md for the full design context.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

# Make repo root importable when pytest is run from anywhere.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _make_variant(query_freq: int = 20, sac_action_chunk_size: int = 1) -> types.SimpleNamespace:
    """Minimal variant for DummyEnv construction. Mirrors what launch_*.py builds."""
    return types.SimpleNamespace(
        env="libero",
        resize_image=64,
        num_cameras=1,
        add_states=1,
        query_freq=query_freq,
        sac_action_chunk_size=sac_action_chunk_size,
    )


# ---------------------------------------------------------------------------
# 1) DummyEnv action_space shape
# ---------------------------------------------------------------------------


def test_dummy_env_action_space_shape_default_unlifted():
    """Default sac_action_chunk_size=1 → reference (1, 32) shape regardless of query_freq."""
    from examples.train_sim_robometer import DummyEnv

    env = DummyEnv(_make_variant(query_freq=20))
    assert env.action_space.shape == (1, 32), env.action_space.shape
    assert env.action_space.low.min() == -1.0
    assert env.action_space.high.max() == 1.0


def test_dummy_env_action_space_shape_lifted():
    from examples.train_sim_robometer import DummyEnv

    env = DummyEnv(_make_variant(query_freq=20, sac_action_chunk_size=20))
    assert env.action_space.shape == (20, 32), env.action_space.shape


def test_dummy_env_rejects_unset_query_freq():
    from examples.train_sim_robometer import DummyEnv

    with pytest.raises(ValueError, match="query_freq"):
        DummyEnv(_make_variant(query_freq=-1))


def test_dummy_env_rejects_invalid_sac_action_chunk_size():
    from examples.train_sim_robometer import DummyEnv

    with pytest.raises(ValueError, match="sac_action_chunk_size"):
        DummyEnv(_make_variant(query_freq=20, sac_action_chunk_size=0))
    with pytest.raises(ValueError, match="sac_action_chunk_size"):
        DummyEnv(_make_variant(query_freq=20, sac_action_chunk_size=51))


def test_dummy_env_other_chunk_sizes():
    from examples.train_sim_robometer import DummyEnv

    env = DummyEnv(_make_variant(query_freq=20, sac_action_chunk_size=5))
    assert env.action_space.shape == (5, 32)
    env = DummyEnv(_make_variant(query_freq=20, sac_action_chunk_size=50))
    assert env.action_space.shape == (50, 32)


# ---------------------------------------------------------------------------
# 2) PixelSACLearner derives action_chunk_shape and action_dim from sample_action
# ---------------------------------------------------------------------------


def test_learner_derives_action_chunk_shape():
    """We don't actually instantiate the learner here (jax + flax init is heavy);
    we verify the math the learner uses at line 136-137 of pixel_sac_learner.py.
    """
    sample_action = np.zeros((1, 20, 32), dtype=np.float32)  # (batch, chunk, latent)
    derived_action_chunk_shape = sample_action.shape[-2:]
    derived_action_dim = int(np.prod(sample_action.shape[-2:]))
    assert derived_action_chunk_shape == (20, 32)
    assert derived_action_dim == 640


# ---------------------------------------------------------------------------
# 3) ReplayBuffer round-trip with the new action shape
# ---------------------------------------------------------------------------


def test_replay_buffer_round_trip_lifted_action():
    from gym.spaces import Box, Dict
    from jaxrl2.data import ReplayBuffer

    obs_space = Dict({
        "pixels": Box(low=0, high=255, shape=(64, 64, 3, 1), dtype=np.uint8),
    })
    action_space = Box(low=-1, high=1, shape=(20, 32), dtype=np.float32)
    buf = ReplayBuffer(obs_space, action_space, capacity=4)
    assert buf.action_space.shape == (20, 32)

    # Insert one transition.
    a0 = np.random.RandomState(0).randn(20, 32).astype(np.float32)
    a1 = np.random.RandomState(1).randn(20, 32).astype(np.float32)
    obs = {"pixels": np.zeros((64, 64, 3, 1), dtype=np.uint8)}
    next_obs = {"pixels": np.ones((64, 64, 3, 1), dtype=np.uint8)}
    buf.insert(dict(
        observations=obs,
        next_observations=next_obs,
        actions=a0,
        next_actions=a1,
        rewards=-1.0,
        masks=1.0,
        discount=0.999 ** 20,
    ))
    assert len(buf) == 1
    np.testing.assert_array_equal(buf.data["actions"][0], a0)
    np.testing.assert_array_equal(buf.data["next_actions"][0], a1)


# ---------------------------------------------------------------------------
# 4) Noise pad-and-concat code path (mirrors collect_traj's i==0 branch)
# ---------------------------------------------------------------------------


def test_noise_build_pad_and_concat_to_pi0_shape():
    """The collect_traj noise-build path must produce (1, 50, 32) regardless of
    the SAC action_chunk_shape. Mirrors lines 743-747 in train_utils_sim_robometer.py.
    """
    action_chunk_shape = (20, 32)
    base_noise = np.random.RandomState(0).randn(1, *action_chunk_shape).astype(np.float32)
    pad_count = 50 - base_noise.shape[1]
    pad = np.repeat(base_noise[:, -1:, :], pad_count, axis=1)
    noise = np.concatenate([base_noise, pad], axis=1)

    assert noise.shape == (1, 50, 32)
    # First 20 rows are the SAC-controlled latents.
    np.testing.assert_array_equal(noise[:, :20], base_noise)
    # Trailing 30 rows all equal the last SAC-controlled row.
    last_row = base_noise[:, -1:, :]
    for k in range(20, 50):
        np.testing.assert_array_equal(noise[:, k:k + 1], last_row)


def test_actor_sample_reshape_round_trip():
    """sample_actions returns (B, action_dim_total). Must reshape to (chunk, latent)
    before the noise-pad path — mirrors line 758 in train_utils_sim_robometer.py.
    """
    action_chunk_shape = (20, 32)
    action_dim = int(np.prod(action_chunk_shape))  # 640
    flat = np.random.RandomState(42).randn(1, action_dim).astype(np.float32)
    reshaped = np.reshape(flat, action_chunk_shape)
    assert reshaped.shape == (20, 32)
    # Round-trip preserves values.
    np.testing.assert_array_equal(reshaped.reshape(1, action_dim), flat)


# ---------------------------------------------------------------------------
# 5) Buffer assertion in add_online_data_to_buffer catches shape regressions
# ---------------------------------------------------------------------------


def test_buffer_assertion_catches_shape_regression():
    from gym.spaces import Box, Dict
    from jaxrl2.data import ReplayBuffer
    from examples.train_utils_sim_robometer import add_online_data_to_buffer

    obs_space = Dict({"pixels": Box(low=0, high=255, shape=(64, 64, 3, 1), dtype=np.uint8)})
    action_space = Box(low=-1, high=1, shape=(20, 32), dtype=np.float32)
    buf = ReplayBuffer(obs_space, action_space, capacity=4)

    # Build a fake traj with the WRONG action chunk shape (1, 32) — pre-lift.
    bad_actions = [np.zeros((1, 32), dtype=np.float32) for _ in range(3)]
    fake_obs = {"pixels": np.zeros((1, 64, 64, 3, 1), dtype=np.uint8)}
    traj = {
        "actions": bad_actions,
        "observations": [fake_obs] * (len(bad_actions) + 1),
    }
    rewards = np.full(len(bad_actions), -1.0, dtype=np.float32)
    masks = np.ones(len(bad_actions), dtype=np.float32)
    variant = types.SimpleNamespace(query_freq=20, discount=0.999, add_states=False)

    with pytest.raises(AssertionError, match="chunk action shape mismatch"):
        add_online_data_to_buffer(variant, traj, rewards, masks, buf)
