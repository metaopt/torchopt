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
    parser = argparse.ArgumentParser([], description='LOLA')

    parser.add_argument('--seed', type=int, default=6666)
    parser.add_argument('--lr_in', type=float, default=0.3, help='Inner Learning rate')

    parser.add_argument('--lr_out', type=float, default=0.2, help='Outer learning rate')
    parser.add_argument('--lr_v', type=float, default=0.1, help='Learning rate of value function')
    parser.add_argument('--gamma', type=float, default=0.96, help='Discount factor')
    parser.add_argument('--n_update', type=int, default=100, help='Number of updates')
    parser.add_argument('--n_lookaheads', type=int, default=1, help='Number of updates')
    parser.add_argument('--len_rollout', type=int, default=150, help='Length of IPD')
    parser.add_argument('--batch_size', type=int, default=1024, help='Natch size')
    parser.add_argument('--use_baseline', action='store_false', default=True)

    args = parser.parse_args()
    return args
