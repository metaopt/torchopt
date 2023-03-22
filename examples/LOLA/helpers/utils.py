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
# https://github.com/alexis-jacq/LOLA_DiCE
# ==============================================================================

import numpy as np
import torch
from torch.distributions import Bernoulli


# evaluate the policy
def step(ipd, theta1, theta2, values1, values2, args):
    # just to evaluate progress:
    (s1, s2), _ = ipd.reset()
    score1 = 0
    score2 = 0
    for _ in range(args.len_rollout):
        a1, lp1, v1 = act(s1, theta1, values1)
        a2, lp2, v2 = act(s2, theta2, values2)
        (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1) / float(args.len_rollout)
        score2 += np.mean(r2) / float(args.len_rollout)
    return (score1, score2)


# dice operator
def magic_box(x):
    return torch.exp(x - x.detach())


# replay buffer
class Memory:
    def __init__(self, args):
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []
        self.args = args

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self, use_baseline=True):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = (
            torch.cumprod(self.args.gamma * torch.ones(*rewards.size()), dim=1) / self.args.gamma
        )
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        if use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(
                torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=1),
            )
            dice_objective = dice_objective + baseline_term

        return -dice_objective  # want to minimize -objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values) ** 2)


def act(batch_states, theta, values):
    batch_states = torch.from_numpy(batch_states).long()
    probs = torch.sigmoid(theta)[batch_states]
    m = Bernoulli(1 - probs)
    actions = m.sample()
    log_probs_actions = m.log_prob(actions)
    return actions.numpy().astype(int), log_probs_actions, values[batch_states]


def sample(ipd, policy, value, args):
    theta1, theta2 = policy
    value1, value2 = value
    (s1, s2), _ = ipd.reset()
    memory_agent1 = Memory(args)
    memory_agent2 = Memory(args)
    for _ in range(args.len_rollout):
        a1, lp1, v1 = act(s1, theta1, value1)
        a2, lp2, v2 = act(s2, theta2, value2)
        (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
        memory_agent1.add(lp1, lp2, v1, torch.from_numpy(r1).float())
        memory_agent2.add(lp2, lp1, v2, torch.from_numpy(r2).float())
    return memory_agent1, memory_agent2
