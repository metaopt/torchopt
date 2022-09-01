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
# This file is modified from:
# https://github.com/tristandeleu/pytorch-maml-rl
# ==============================================================================

import torch.nn as nn
from torch.distributions import Categorical


class CategoricalMLPPolicy(nn.Module):
    """Policy network based on a multi-layer perceptron (MLP), with a
    `Categorical` distribution output. This policy network can be used on tasks
    with discrete action spaces (eg. `TabularMDPEnv`).
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(32, output_size)
        self.value_head = nn.Linear(32, 1)

    def forward(self, inputs, params=None):
        embedding = self.torso(inputs)
        logits = self.policy_head(embedding)
        values = self.value_head(embedding)
        return Categorical(logits=logits), values
