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
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)

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
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    