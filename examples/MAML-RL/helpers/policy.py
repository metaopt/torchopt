# This file is modified from:
# https://github.com/tristandeleu/pytorch-maml-rl

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchrl.modules import ActorValueOperator, ValueOperator, ProbabilisticActor, OneHotCategorical, TDModule

class Backbone(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 ):
        super(Backbone, self).__init__()
        self.torso = nn.Sequential(nn.Linear(input_size, 32), nn.ReLU(),
                               nn.Linear(32, 32), nn.ReLU(),
                               )
        self.policy_head = nn.Linear(32, output_size)

    def forward(self, inputs, params=None):
        embedding = self.torso(inputs)
        return embedding

class CategoricalSubNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 ):
        super(CategoricalSubNet, self).__init__()
        self.policy_head = nn.Linear(32, output_size)

    def forward(self, embedding, params=None):
        logits = self.policy_head(embedding)
        return logits

class ValueSubNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 ):
        super(CategoricalSubNet, self).__init__()
        self.value_head = nn.Linear(32, 1)

    def forward(self, embedding, params=None):
        value = self.value_head(embedding)
        return value

class ActorCritic(ActorValueOperator):
    def __init__(self, input_size, output_size):
        super().__init__(
            TDModule(
                module=Backbone(input_size, output_size),
                in_keys=["observation"],
                out_keys=["hidden"],
            ),
            ProbabilisticActor(
                module=CategoricalSubNet(input_size, output_size),
                distribution_class=OneHotCategorical,
                return_log_prob=False,
                in_keys=["hidden"],
                out_keys=["action"],
                default_interaction_mode="random",
            ),
            ValueOperator(
                module=ValueSubNet(input_size, output_size),
                in_keys=["hidden"],
            )
        )