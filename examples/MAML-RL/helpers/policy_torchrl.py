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


import torch
import torch.nn as nn
from torch.distributions import Categorical
from torchrl.modules import (
    ActorValueOperator,
    OneHotCategorical,
    ProbabilisticActor,
    TensorDictModule,
    ValueOperator,
)


class Backbone(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super(Backbone, self).__init__()
        self.torso = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

    def forward(self, inputs, params=None):
        embedding = self.torso(inputs)
        return embedding


class CategoricalSubNet(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super().__init__()
        self.policy_head = nn.Linear(32, output_size)

    def forward(self, embedding, params=None):
        logits = self.policy_head(embedding)
        return logits


class ValueSubNet(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super().__init__()
        self.value_head = nn.Linear(32, 1)

    def forward(self, embedding, params=None):
        value = self.value_head(embedding)
        return value


class ActorCritic(ActorValueOperator):
    def __init__(self, input_size, output_size):
        super().__init__(
            TensorDictModule(
                spec=None,
                module=Backbone(input_size, output_size),
                in_keys=['observation'],
                out_keys=['hidden'],
            ),
            ProbabilisticActor(
                spec=None,
                module=TensorDictModule(
                    CategoricalSubNet(input_size, output_size),
                    in_keys=['hidden'],
                    out_keys=['logits'],
                ),
                distribution_class=OneHotCategorical,
                return_log_prob=False,
                dist_param_keys=['logits'],
                out_key_sample=['action'],
            ),
            ValueOperator(
                module=ValueSubNet(input_size, output_size),
                in_keys=['hidden'],
            ),
        )
