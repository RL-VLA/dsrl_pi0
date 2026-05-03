#! /usr/bin/env python
import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs from https://github.com/huggingface/gym-aloha/tree/main?tab=readme-ov-file#-gpu-rendering-egl
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib, copy

import jax
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np

import gymnasium as gym
import gym_aloha
from gym.spaces import Dict, Box

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from jaxrl2.data import ReplayBuffer
from jaxrl2.data.dsrl_na_replay_buffer import DSRLNAReplayBuffer
from jaxrl2.agents.pixel_dsrl_na import PixelDSRLNALearner
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_sim_robometer import trajwise_alternating_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache

from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download

home_dir = os.environ['HOME']
compilation_cache.set_cache_dir(os.path.join(home_dir, 'jax_compilation_cache'))

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax NamedSharding partitioning the leading axis across devices.
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, sharding),
        batch,
    )


class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            if variant.env == 'libero':
                state_dim = 8
            elif variant.env == 'aloha_cube':
                state_dim = 14
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        # SAC noise action shape: (sac_action_chunk_size, 32). Decoupled from
        # query_freq so we can choose between the reference behaviour
        # (sac_action_chunk_size=1, single latent broadcast across pi0's
        # 50-step noise tensor) and the lifted form (sac_action_chunk_size>1,
        # one latent per pi0 noise row, trailing rows filled by repeating the
        # last SAC row downstream in collect_traj). See
        # docs/sac_action_lift_plan.md.
        if not (isinstance(variant.query_freq, int) and variant.query_freq > 0):
            raise ValueError(
                f"variant.query_freq must be a positive int (got {variant.query_freq!r}); "
                "set --query_freq on the launcher."
            )
        sac_chunk = int(getattr(variant, 'sac_action_chunk_size', 1))
        if sac_chunk < 1 or sac_chunk > 50:
            raise ValueError(
                f"sac_action_chunk_size must be in [1, 50] (pi0 action_chunk_size); "
                f"got {sac_chunk}."
            )
        self.action_space = Box(
            low=-1, high=1,
            shape=(sac_chunk, 32),
            dtype=np.float32,
        )
        # Diffused env-action space — used by DSRL-NA's action critic. The
        # last-axis dim is the env's actuator count (libero=7, aloha=14). The
        # SAC actor never sees this space; only the diffused-action critic does.
        if variant.env == 'libero':
            env_action_dim = 7
        elif variant.env == 'aloha_cube':
            env_action_dim = 14
        else:
            raise NotImplementedError(f"diffused action_dim unknown for env={variant.env!r}")
        self.diffused_action_space = Box(
            low=-1, high=1,
            shape=(variant.query_freq, env_action_dim),
            dtype=np.float32,
        )


def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    print('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    mesh = jax.sharding.Mesh(np.array(devices), ('batch',))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    if variant.env == 'libero':
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict["libero_90"]()
        task_id = 57
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(task, 256, variant.seed)
        eval_env = env
        variant.task_description = task_description
        variant.env_max_reward = 1
        variant.max_timesteps = 400
    elif variant.env == 'aloha_cube':
        from gymnasium.envs.registration import register
        register(
            id="gym_aloha/AlohaTransferCube-v0",
            entry_point="gym_aloha.env:AlohaEnv",
            max_episode_steps=400,
            nondeterministic=True,
            kwargs={"obs_type": "pixels", "task": "transfer_cube"},
        )
        env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
        eval_env = copy.deepcopy(env)
        variant.env_max_reward = 4
        variant.max_timesteps = 400
        

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    print('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    print('sample action shape', sample_action.shape)
    

    if variant.env == 'libero':
        config = openpi_config.get_config("pi0_libero")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
    elif variant.env == 'aloha_cube':
        config = openpi_config.get_config("pi0_aloha_sim")
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
    else:
        raise NotImplementedError()
    agent_dp = policy_config.create_trained_policy(config, checkpoint_dir)
    flow_steps = int(getattr(variant, 'flow_integration_steps', -1))
    if flow_steps > 0 and hasattr(agent_dp, '_sample_kwargs'):
        agent_dp._sample_kwargs['num_steps'] = flow_steps
        print(f"pi0 flow_integration_steps overridden to {flow_steps}")
    print("Loaded pi0 policy from %s", checkpoint_dir)

    algorithm = getattr(variant, "algorithm", "pixel_sac")
    cli_buf = int(getattr(variant, 'online_buffer_size', -1))
    if cli_buf > 0:
        online_buffer_size = cli_buf
    else:
        online_buffer_size = variant.max_steps // variant.multi_grad_step
    if algorithm == "pixel_sac":
        agent = PixelSACLearner(variant.seed, sample_obs, sample_action, **kwargs)
        online_replay_buffer = ReplayBuffer(
            dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size)
        )
    elif algorithm == "pixel_dsrl_na":
        sample_diffused_action = add_batch_dim(dummy_env.diffused_action_space.sample())
        print('sample diffused action shape', sample_diffused_action.shape)
        bup_ent = bool(int(getattr(variant, "dsrl_na_backup_entropy", 0)))
        noise_scale_inside = bool(int(getattr(variant, "noise_scale_inside", 0)))
        agent = PixelDSRLNALearner(
            variant.seed,
            observations=sample_obs,
            actions=sample_action,                 # noise actions, shape (1, sac_action_chunk_size, 32)
            env_actions=sample_diffused_action,    # diffused env-actions, shape (1, query_freq, 7)
            backup_entropy=bup_ent,
            noise_scale_inside=noise_scale_inside,
            **kwargs,
        )
        # Buffer's executed_action_dim is the FLAT env-action dim (e.g. 20*7=140 for libero).
        executed_action_dim = int(np.prod(dummy_env.diffused_action_space.shape))
        online_replay_buffer = DSRLNAReplayBuffer(
            dummy_env.observation_space,
            dummy_env.action_space,
            executed_action_dim=executed_action_dim,
            capacity=int(online_buffer_size),
        )
    else:
        raise ValueError(f"unknown variant.algorithm={algorithm!r}")
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp)
 