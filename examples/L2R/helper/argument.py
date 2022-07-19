# Copyright 2022 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse


def parse_args():
    parser = argparse.ArgumentParser([], description='L2R')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=30, help='Training Epoch')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument(
        '--pos_ratio',
        type=float,
        default=0.995,
        help='Ratio of positive examples in training',
    )
    parser.add_argument('--ntest', type=int, default=500, help='Number of testing examples')
    parser.add_argument('--ntrain', type=int, default=5000, help='Number of testing examples')
    parser.add_argument('--nval', type=int, default=10, help='Number of valid examples')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')

    # For baseline
    parser.add_argument('--algo', type=str, default='both')

    args = parser.parse_args()
    # use the GPU if available
    return args
