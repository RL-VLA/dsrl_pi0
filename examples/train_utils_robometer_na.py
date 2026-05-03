"""Shared training utilities for DSRL-NA."""
from jaxrl2.agents.agent import get_batch_stats
from jaxrl2.agents.common import sample_actions_seeded_jit
import time
from functools import partial
from tqdm import tqdm
import numpy as np
import jax
import jax.numpy as jnp
from flax.core import frozen_dict
from termcolor import colored


def remove_original_obs_keys(batch):
    """Remove original observation keys from batch before passing to JIT-compiled update.
    
    These keys contain non-JIT-compatible data (e.g., string prompts) and are only
    needed for generate_distillation_batch(), not for the agent update itself.
    """
    keys_to_remove = ['original_observations', 'original_next_observations',
                    'original_k_cache', 'original_v_cache',
                    'original_next_k_cache', 'original_next_v_cache']
    batch_dict = dict(batch)
    for key in keys_to_remove:
        batch_dict.pop(key, None)
    return frozen_dict.freeze(batch_dict)

def get_distillation_actions_from_dp(agent_dp, batch, actions_noise, action_chunk_shape, robot_config, variant):
    times = {}
    batch_size = actions_noise.shape[0]
    chunk, dim = action_chunk_shape
    T = robot_config["action_chunk_size"]
    start = time.time()
    obs_for_dp = batch['original_observations']
    times['obs_prep_time'] = time.time() - start
    
    start = time.time()
    # noise_for_dp = np.concatenate(noise_for_dp, axis=0) # (batch_size, chunk_size, noise_dim)
    noise = actions_noise.reshape(batch_size, chunk, dim) # (B, chunk, dim)
    idx = jnp.minimum(jnp.arange(T), chunk - 1)           # (T,)
    noise_for_dp = noise[:, idx, :]                       # (B, T, dim)
    times['noise_concat_time'] = time.time() - start
    
    # k_cache_list.shape = (batch_size, 18, 1, num_tokens, 1, 256)
    # Convert to (18, batch_size, num_tokens, 1, 256) for DP input
    start = time.time()
    kv_cache_for_dp = None
    if variant['save_kv_cache']:
        k_cache_list = batch['original_k_cache']
        v_cache_list = batch['original_v_cache']
        k_cache_batch = jax.numpy.concatenate(k_cache_list, axis=1)
        v_cache_batch = jax.numpy.concatenate(v_cache_list, axis=1)
        kv_cache_for_dp = (k_cache_batch, v_cache_batch)
    times['kv_cache_concat_time'] = time.time() - start
    
    start = time.time()
    agent_dp_output = agent_dp.infer_batch(obs_for_dp, noise=noise_for_dp, kv_cache=kv_cache_for_dp)['actions']
    times['dp_time'] = time.time() - start
    agent_dp_actions = agent_dp_output[..., :variant.query_freq, :]
    agent_dp_actions = agent_dp_actions.reshape(batch_size, -1)
    return agent_dp_actions, times

def get_next_actions_from_dp(agent_dp, batch, next_actions_noise, action_chunk_shape, robot_config, variant):
    times = {}
    batch_size = next_actions_noise.shape[0]
    start = time.time()
    next_obs_for_dp = batch['original_next_observations']
    o_times = time.time() - start
    next_noise_for_dp = []
    n_times = 0
    for i in range(batch_size):
        start = time.time()
        noise_i = next_actions_noise[i].reshape(1, action_chunk_shape[0], action_chunk_shape[1])
        noise_repeat = np.repeat(noise_i[:, -1:, :], robot_config['action_chunk_size'] - noise_i.shape[1], axis=1)
        noise_full = np.concatenate([noise_i, noise_repeat], axis=1)
        next_noise_for_dp.append(noise_full)
        n_times += time.time() - start
    
    times['noise_prep_time'] = n_times
    times['obs_prep_time'] = o_times
    start = time.time()
    next_noise_for_dp = np.concatenate(next_noise_for_dp, axis=0) # (batch_size, chunk_size, noise_dim)
    times['noise_concat_time'] = time.time() - start
    # kv_cache_for_dp.shape = (batch_size, 18, 1, num_tokens, 1, 256)
    # Convert to (18, batch_size, num_tokens, 1, 256) for DP input
    start = time.time()
    next_kv_cache_for_dp = None
    if variant['save_kv_cache']:
        next_k_cache_list = batch['original_next_k_cache']
        next_v_cache_list = batch['original_next_v_cache']
        next_k_cache_batch = jax.numpy.concatenate(next_k_cache_list, axis=1)
        next_v_cache_batch = jax.numpy.concatenate(next_v_cache_list, axis=1)
        next_kv_cache_for_dp = (next_k_cache_batch, next_v_cache_batch)
    times['kv_cache_concat_time'] = time.time() - start
    
    start = time.time()
    agent_dp_output = agent_dp.infer_batch(next_obs_for_dp, noise=next_noise_for_dp, kv_cache=next_kv_cache_for_dp)['actions']
    times['dp_time'] = time.time() - start
    # agent_dp_output.shape = (batch_size, action_horizon, env_action_dim).
    # Slice the action_horizon (axis -2) to query_freq.
    agent_dp_next_actions = agent_dp_output[..., :variant.query_freq, :]
    agent_dp_next_actions = agent_dp_next_actions.reshape(batch_size, -1)
    return agent_dp_next_actions, times

@partial(jax.jit, static_argnames=("noise_dim", "grl_noise_sample"))
def choose_noise(rng, actor, obs_batch, noise_dim, grl_noise_sample):
    batch_size = obs_batch["pixels"].shape[0]
    if not grl_noise_sample:
        rng, k = jax.random.split(rng)
        return rng, jax.random.normal(k, (batch_size, noise_dim))

    grl_noise_p=0.5
    rng, k_choose, k_normal, k_actor = jax.random.split(rng, 4)
    use_normal = jax.random.bernoulli(k_choose, grl_noise_p)  # scalar

    actor_batch_stats = get_batch_stats(actor)

    def normal_branch(_):
        return jax.random.normal(k_normal, (batch_size, noise_dim))

    def actor_branch(_):
        return sample_actions_seeded_jit(
            k_actor,
            actor.apply_fn,
            actor.params,
            obs_batch,
            actor_batch_stats,
        )

    noise = jax.lax.cond(use_normal, normal_branch, actor_branch, operand=None)
    return rng, noise

def generate_distillation_batch(batch, agent, agent_dp, robot_config, variant, agent_dp_device=None):
    """Generate distillation data by querying diffusion policy with random noise."""
    batch_size = batch['observations']['pixels'].shape[0]
    noise_dim = agent.action_dim
    
    t0 = time.time()
    agent._rng, distill_noise = choose_noise(agent._rng, agent._actor, batch["observations"], noise_dim, variant['grl_noise_sample'])
    choose_noise_time = time.time() - t0
    if agent_dp_device is not None:
        distill_noise = jax.device_put(distill_noise, agent_dp_device)
    distill_actions, end_times = get_distillation_actions_from_dp(agent_dp, batch, distill_noise, agent.action_chunk_shape, robot_config, variant)
    end_times['choose_noise_time'] = choose_noise_time
    return distill_noise, distill_actions, end_times

def add_online_data_to_buffer_na(variant, traj, online_replay_buffer):
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions'])
    executed_actions = np.array(traj['executed_actions'])
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])
    k_caches = traj.get('original_k_cache', None)
    v_caches = traj.get('original_v_cache', None)
    if k_caches is None or v_caches is None:
        tqdm.write(colored("Not logging the k_cache and v_cache to the buffer", "red"))

    executed_actions_flat = executed_actions.reshape(episode_len, -1)
    
    for t in range(episode_len):
        obs = traj['observations'][t]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
        
        # Get original observation for DP queries during distillation
        original_obs = traj['original_observations'][t]
        
        # Note: next_observations, next_actions are derived
        # at sample time by ReplayBufferNA, so we don't store them here.
        insert_dict = dict(
            observations=obs,
            actions=actions[t],
            executed_actions=executed_actions_flat[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon,
            original_observations=original_obs,
        )

        # Add kv_cache if available
        if k_caches is not None and v_caches is not None:
            insert_dict['original_k_cache'] = k_caches[t]
            insert_dict['original_v_cache'] = v_caches[t]
        
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()
