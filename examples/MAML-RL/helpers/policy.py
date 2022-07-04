# This file is modified from:
# https://github.com/tristandeleu/pytorch-maml-rl

import torch
import torch.nn as nn
from torch.distributions import Categorical


class CategoricalMLPPolicy(nn.Module):
    """Policy network based on a multi-layer perceptron (MLP), with a 
    `Categorical` distribution output. This policy network can be used on tasks 
    with discrete action spaces (eg. `TabularMDPEnv`). 
    """
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super(CategoricalMLPPolicy, self).__init__()
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
