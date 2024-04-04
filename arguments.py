import argparse

"""
Here are the param for the training
"""

# bandu
# pusher, pointmass_rooms
# kitchenSeq
# pusher_hard



def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='pointmass_rooms', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=20001, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=1, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=100, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--seed', type=int, default=66, help='random seed')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise eps')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')
    parser.add_argument('--meta-batch-size', type=int, default=256, help='the sample meta batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-upper-actor', type=float, default=3e-4, help='the learning rate of the upper actor')
    parser.add_argument('--lr-upper-critic', type=float, default=3e-4, help='the learning rate of the upper critic')
    parser.add_argument('--lr-reward-model', type=float, default=3e-4, help='the learning rate of the reward model')
    parser.add_argument('--lr-rnd-model', type=float, default=3e-4, help='the learning rate of the rnd model')
    parser.add_argument('--lr-alpha-model', type=float, default=10e-4, help='the learning rate of the alpha model')
    parser.add_argument('--lr-actor', type=float, default=3e-4, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=3e-4, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=50, help='the number of tests')
    parser.add_argument('--test-interval', type=int, default=50, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', action='store_true', default=True, help='if use gpu do the acceleration')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=1, help='the rollouts per mpi')

    parser.add_argument('--feed_type', type=int, default=0, help='the feedback type')
    parser.add_argument('--reward_update', type=int, default=200, help='the rollouts per mpi')
    parser.add_argument('--reward_batch', type=int, default=25, help='the rollouts per mpi')
    parser.add_argument('--label_margin', type=int, default=0, help='the rollouts per mpi')

    parser.add_argument('--num_seed_steps', type=int, default=500, help='the rollouts per mpi')
    parser.add_argument('--num_unsup_steps', type=int, default=1000, help='the rollouts per mpi')
    parser.add_argument('--num_interact', type=int, default=200, help='the rollouts per mpi')
    parser.add_argument('--reward_schedule', type=int, default=0, help='the rollouts per mpi')
    parser.add_argument('--num_train_steps', type=int, default=1e6, help='the rollouts per mpi')
    parser.add_argument('--segment', type=int, default=1, help='the rollouts per mpi')
    parser.add_argument('--max_feedback', type=int, default=10025, help='the rollouts per mpi')

    parser.add_argument('--teacher_eps_mistake', type=int, default=0, help='the rollouts per mpi')
    parser.add_argument('--teacher_eps_skip', type=int, default=0, help='the rollouts per mpi')
    parser.add_argument('--teacher_eps_equal', type=int, default=0, help='the rollouts per mpi')

    parser.add_argument('--log_frequency', type=int, default=1, help='the rollouts per mpi')
    parser.add_argument('--log_save_tb', type=bool, default=True, help='the rollouts per mpi')


    args = parser.parse_args()

    return args
