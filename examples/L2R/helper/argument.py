import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser([], description='L2R')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=30, help='Training Epoch')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--pos_ratio',
                        type=float,
                        default=0.995,
                        help='Ratio of positive examples in training')
    parser.add_argument('--ntest',
                        type=int,
                        default=500,
                        help='Number of testing examples')
    parser.add_argument('--ntrain',
                        type=int,
                        default=5000,
                        help='Number of testing examples')
    parser.add_argument('--nval',
                        type=int,
                        default=10,
                        help='Number of valid examples')
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='Batch size')

    ### For baseline
    parser.add_argument('--algo', type=str, default='both')

    args = parser.parse_args()
    # use the GPU if available
    return args
