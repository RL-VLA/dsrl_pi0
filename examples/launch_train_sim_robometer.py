import argparse
import sys
from examples.train_sim_robometer import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac',
                        choices=['pixel_sac', 'pixel_dsrl_na'],
                        help='pixel_sac = single noise critic (DSRL-SAC); '
                             'pixel_dsrl_na = two-critic DSRL-NA (action critic + '
                             'distilled noise critic). See docs/dsrl_na_integration_plan.md.')
    parser.add_argument('--noise_critic_grad_steps', default=1, type=int,
                        help='[pixel_dsrl_na only] inner distillation steps per '
                             'outer SAC update. Reference impl uses 10. (Currently '
                             'unused in v0 — kept for forward compat.)')
    parser.add_argument('--dsrl_na_backup_entropy', default=0, type=int,
                        choices=[0, 1],
                        help='[pixel_dsrl_na only] include the soft-actor-critic '
                             'entropy bonus in the action critic Bellman target. '
                             'Reference jaxrl2 DSRL-NA defaults to 0; the SB3 fork '
                             'always has it on. Either way, the actor still uses '
                             'entropy in its loss.')
    parser.add_argument('--save_kv_cache', default=0, type=int, choices=[0, 1],
                        help='[pixel_dsrl_na only] cache pi0 prompt-encoder K/V '
                             'per buffer slot at rollout time so per-update pi0 '
                             'forwards reuse it. Cuts pi0 cost meaningfully; needs '
                             'the arayabrain openpi fork.')
    parser.add_argument('--grl_noise_sample', default=0, type=int, choices=[0, 1],
                        help='[pixel_dsrl_na only] distillation noise sampler. '
                             '0 = always fresh N(0, I); 1 = mix of N(0, I) and '
                             'actor-sampled noise (50/50). The reference uses 1 '
                             'when the actor has converged to a tight distribution.')
    parser.add_argument('--noise_scale_inside', default=1, type=int, choices=[0, 1],
                        help='[pixel_dsrl_na only] 1 = bound the actor distribution '
                             'to [-action_magnitude, +action_magnitude] internally '
                             '(matches our DSRL-SAC). 0 = bound to [-1, 1] then '
                             'multiply by action_magnitude externally (matches the '
                             'arayabrain reference default).')
    parser.add_argument('--train_action_critic', default=1, type=int, choices=[0, 1],
                        help='[pixel_dsrl_na only] toggle action-critic updates. '
                             '0 disables for ablations.')
    parser.add_argument('--train_noise_actor', default=1, type=int, choices=[0, 1],
                        help='[pixel_dsrl_na only] toggle noise-actor updates.')
    parser.add_argument('--train_noise_critic', default=1, type=int, choices=[0, 1],
                        help='[pixel_dsrl_na only] toggle noise-critic distillation.')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--sac_action_chunk_size', default=1, type=int,
                        help='Number of 32-d noise latents the SAC actor outputs '
                             'per chunk. 1 = reference DSRL/DSRL-NA behaviour '
                             '(single latent broadcast across pi0\'s 50-step '
                             'noise tensor). >1 = lifted: SAC controls one latent '
                             'per row of pi0\'s diffusion noise; trailing rows '
                             'are filled by repeating the last SAC-controlled '
                             'row. Must be in [1, action_chunk_size=50]. Set to '
                             'query_freq to make every executed env-step have '
                             'its own SAC latent. See docs/sac_action_lift_plan.md.')

    # ---- Robometer dense-reward plumbing ------------------------------------
    parser.add_argument('--robometer_url', default='http://localhost:8000',
                        help='URL of the Robometer eval server.')
    parser.add_argument('--robometer_reward_kind', default='libero_success_plus_robo_progress',
                        choices=['libero_success_plus_robo_progress',
                                 'robo_success_plus_robo_progress',
                                 'robo_success_sparse',
                                 'progress_delta', 'progress', 'success', 'libero_sparse'],
                        help='Reward function applied to the robometer scores.')
    parser.add_argument('--robometer_success_threshold', default=0.5, type=float,
                        help='Threshold on robometer success prob to flag an episode as '
                             'successful (only used by robo_success_* reward fns).')
    parser.add_argument('--robometer_frame_size', default=224, type=int,
                        help='Spatial resolution sent to robometer (frames are resized).')
    parser.add_argument('--robometer_use_frame_steps', default=1, type=int,
                        help='1 = server runs one forward per prefix (dense causal progress); '
                             '0 = single forward per video.')
    parser.add_argument('--robometer_fail_behavior', default='raise',
                        choices=['raise', 'drop'],
                        help='raise = crash on server error (loudest, easiest to debug); '
                             'drop = discard the episode and keep training. No silent '
                             'reward-fn fallback — a reward function never substitutes '
                             'itself for another.')
    parser.add_argument('--robometer_queue_max_depth', default=4, type=int,
                        help='Max in-flight episode-scoring requests before we block on the oldest.')
    parser.add_argument('--robometer_max_retries', default=3, type=int,
                        help='HTTP retries inside the RobometerClient for transient errors.')

    # ---- Reference-impl alignment knobs ------------------------------------
    parser.add_argument('--target_entropy', default='auto',
                        help="SAC target entropy. 'auto' = -action_dim (our "
                             'default). Set to a float like 0.0 to match the '
                             'arayabrain reference DSRL-NA.')
    parser.add_argument('--flow_integration_steps', default=-1, type=int,
                        help='Number of pi0 diffusion (flow integration) steps '
                             "per inference. -1 = use pi0's default. Reference "
                             'DSRL-NA uses 10.')
    parser.add_argument('--online_buffer_size', default=-1, type=int,
                        help='Replay buffer capacity. -1 = auto '
                             '(max_steps/multi_grad_step). Reference uses '
                             '150_000 regardless of max_steps.')
    parser.add_argument('--num_initial_traj_collect', default=-1, type=int,
                        help='Number of base-policy trajectories to collect '
                             'before SAC updates begin. -1 = derive from '
                             '--start_online_updates. Reference uses 125 '
                             '(aloha) for warmup.')
    parser.add_argument('--action_critic_steps', default=1, type=int,
                        help='[pixel_dsrl_na only] inner action-critic '
                             'gradient steps per outer SAC step. Reference '
                             'uses 15.')
    parser.add_argument('--noise_critic_steps', default=1, type=int,
                        help='[pixel_dsrl_na only] inner noise-critic '
                             'distillation steps per outer SAC step. Reference '
                             'uses 5. Each step draws a fresh distillation '
                             'noise + pi0 forward (or reuses K/V cache).')
    parser.add_argument('--noise_actor_steps', default=1, type=int,
                        help='[pixel_dsrl_na only] inner actor gradient steps '
                             'per outer SAC step. Reference uses 5.')
    parser.add_argument('--put_kv_cache_on_cpu', default=0, type=int, choices=[0, 1],
                        help='[pixel_dsrl_na + save_kv_cache only] place pi0 '
                             'K/V cache tensors in host memory instead of '
                             'GPU. Saves substantial GPU memory at high buffer '
                             'capacities (reference default at '
                             'online_buffer_size=150_000). Pages onto GPU only '
                             'when pi0 needs them at update time.')

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    # Parse target_entropy: 'auto' string or numeric float, then inject into
    # train_kwargs so it reaches the SAC learner constructor.
    te_raw = variant.target_entropy
    if isinstance(te_raw, str) and te_raw.strip().lower() == 'auto':
        te_value = 'auto'
    else:
        try:
            te_value = float(te_raw)
        except (TypeError, ValueError):
            raise ValueError(f"--target_entropy must be 'auto' or a float, got {te_raw!r}")
    variant.target_entropy = te_value
    variant['train_kwargs']['target_entropy'] = te_value
    print(variant)
    main(variant)
    sys.exit()
    