import argparse


def parse_args():
    parser = argparse.ArgumentParser([], description='LOLA')

    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--lr_in',
                        type=float,
                        default=0.3,
                        help='Inner Learning rate')

    parser.add_argument('--lr_out',
                        type=float,
                        default=0.2,
                        help='Outer learning rate')
    parser.add_argument('--lr_v',
                        type=float,
                        default=0.1,
                        help='Learning rate of value function')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.96,
                        help='Discount factor')
    parser.add_argument('--n_update',
                        type=int,
                        default=100,
                        help='Number of updates')
    parser.add_argument('--n_lookaheads',
                        type=int,
                        default=1,
                        help='Number of updates')
    parser.add_argument('--len_rollout',
                        type=int,
                        default=150,
                        help='Length of IPD')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='Natch size')
    parser.add_argument('--use_baseline', action='store_false', default=True)

    args = parser.parse_args()
    return args
