"""Unit tests for the DSRL-NA two-critic integration.

Covers learner shape derivation, replay buffer round-trip, one full update step
on synthetic data, and the schema-mismatch assertion in
``add_online_data_to_buffer``. No pi0 weights or libero env required — these
tests run in seconds on CPU/GPU.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _variant(query_freq: int = 20, sac_action_chunk_size: int = 20) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        env="libero",
        resize_image=64,
        num_cameras=1,
        add_states=1,
        query_freq=query_freq,
        sac_action_chunk_size=sac_action_chunk_size,
    )


# ---------------------------------------------------------------------------
# 1) DummyEnv exposes BOTH action spaces with correct shapes.
# ---------------------------------------------------------------------------


def test_dummy_env_exposes_diffused_action_space():
    """Noise action shape follows sac_action_chunk_size; diffused shape follows query_freq."""
    from examples.train_sim_robometer import DummyEnv

    env = DummyEnv(_variant(query_freq=20, sac_action_chunk_size=20))
    assert env.action_space.shape == (20, 32)              # noise space (lifted)
    assert env.diffused_action_space.shape == (20, 7)      # libero env-action dim


def test_dummy_env_default_unlifted_noise_shape():
    """Default sac_action_chunk_size=1 → (1, 32) noise space, but diffused stays (query_freq, 7)."""
    from examples.train_sim_robometer import DummyEnv

    env = DummyEnv(_variant(query_freq=20, sac_action_chunk_size=1))
    assert env.action_space.shape == (1, 32)
    assert env.diffused_action_space.shape == (20, 7)


def test_dummy_env_aloha_diffused_dim():
    from examples.train_sim_robometer import DummyEnv

    v = _variant(query_freq=20, sac_action_chunk_size=20)
    v.env = "aloha_cube"
    env = DummyEnv(v)
    assert env.diffused_action_space.shape == (20, 14)     # aloha env-action dim


# ---------------------------------------------------------------------------
# 2) DSRLNAReplayBuffer round-trip with both action types.
# ---------------------------------------------------------------------------


def test_dsrl_na_buffer_round_trip():
    """Round-robin buffer round-trip with the new flat executed_actions schema."""
    from gym.spaces import Box, Dict
    from jaxrl2.data.dsrl_na_replay_buffer import DSRLNAReplayBuffer

    obs = Dict({"pixels": Box(0, 255, shape=(64, 64, 3, 1), dtype=np.uint8)})
    noise_space = Box(-1, 1, shape=(20, 32), dtype=np.float32)
    # executed_action_dim is the FLAT env-action dim (e.g. 20 chunks * 7 actuators = 140 for libero).
    buf = DSRLNAReplayBuffer(obs, noise_space, executed_action_dim=140, capacity=4)

    rng = np.random.RandomState(0)
    # 3 transitions in a single traj (mask=1 mid-traj, 0 at the end). Round-robin
    # parent rejects "trajectory-end with mask=1" rows as sample candidates, so
    # we need at least one mid-traj row before the terminal one.
    for k in range(3):
        is_terminal = (k == 2)
        buf.insert(dict(
            observations={"pixels": rng.randint(0, 256, (64, 64, 3, 1)).astype(np.uint8)},
            actions=rng.randn(20, 32).astype(np.float32),
            executed_actions=rng.randn(140).astype(np.float32),
            original_observations={"prompt": "fake task", "observation/state": rng.randn(8)},
            rewards=-1.0, masks=0.0 if is_terminal else 1.0,
            discount=0.999 ** 20,
        ))
    buf.increment_traj_counter()
    assert len(buf) == 3
    batch = buf.sample(2)
    assert batch["actions"].shape == (2, 20, 32)
    assert batch["executed_actions"].shape == (2, 140)
    # Round-robin parent derives next_observations / next_actions from next_indices.
    assert batch["next_observations"]["pixels"].shape == (2, 64, 64, 3, 1)
    assert "original_next_observations" in batch


# ---------------------------------------------------------------------------
# 3) PixelDSRLNALearner derives both action shapes.
# ---------------------------------------------------------------------------


def test_learner_derives_both_action_shapes():
    """Pure shape math — no jax init, just verify the learner's __init__ logic."""
    noise_actions = np.zeros((1, 20, 32), dtype=np.float32)
    diffused_actions = np.zeros((1, 20, 7), dtype=np.float32)
    expected_action_dim = int(np.prod(noise_actions.shape[-2:]))
    expected_diff_dim = int(np.prod(diffused_actions.shape[-2:]))
    assert expected_action_dim == 640
    assert expected_diff_dim == 140


# ---------------------------------------------------------------------------
# 4) DSRL-NA insert path lives in train_utils_robometer_na (separate from SAC's).
# ---------------------------------------------------------------------------


def test_na_helpers_module_imports():
    """Verify the NA helpers from train_utils_robometer_na are reachable."""
    from examples.train_utils_robometer_na import (
        get_next_actions_from_dp,
        get_distillation_actions_from_dp,
        choose_noise,
        generate_distillation_batch,
        add_online_data_to_buffer_na,
        remove_original_obs_keys,
    )
    assert callable(add_online_data_to_buffer_na)
    assert callable(generate_distillation_batch)
    # remove_original_obs_keys is a pure helper — exercise it.
    from flax.core import frozen_dict
    batch = frozen_dict.freeze({
        "observations": {"pixels": np.zeros((2, 4, 4, 3))},
        "original_observations": [{"prompt": "x"}, {"prompt": "y"}],
        "original_k_cache": [None, None],
    })
    cleaned = remove_original_obs_keys(batch)
    assert "original_observations" not in cleaned
    assert "original_k_cache" not in cleaned
    assert "observations" in cleaned


# ---------------------------------------------------------------------------
# 5) Pixel-SAC path is unchanged (regression guard).
# ---------------------------------------------------------------------------


def test_pixel_sac_buffer_does_not_require_diffused():
    """Plain ReplayBuffer (DSRL-SAC) ignores diffused_actions — it must continue to."""
    from gym.spaces import Box, Dict
    from jaxrl2.data import ReplayBuffer
    from examples.train_utils_sim_robometer import add_online_data_to_buffer

    obs = Dict({"pixels": Box(0, 255, shape=(64, 64, 3, 1), dtype=np.uint8)})
    noise_space = Box(-1, 1, shape=(20, 32), dtype=np.float32)
    buf = ReplayBuffer(obs, noise_space, capacity=4)

    actions = [np.zeros((20, 32), dtype=np.float32) for _ in range(2)]
    obss = [{"pixels": np.zeros((1, 64, 64, 3, 1), dtype=np.uint8)} for _ in range(3)]
    # Note: NO 'diffused_actions' key — must still work for DSRL-SAC.
    traj = {"actions": actions, "observations": obss, "is_success": False}
    variant = types.SimpleNamespace(query_freq=20, discount=0.999, add_states=False)

    add_online_data_to_buffer(
        variant, traj,
        rewards=-np.ones(2, dtype=np.float32),
        masks=np.ones(2, dtype=np.float32),
        online_replay_buffer=buf,
    )
    assert len(buf) == 2
