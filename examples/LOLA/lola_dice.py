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

from helpers.agent import Agent
from helpers.argument import parse_args
from helpers.env import IPD
from helpers.utils import sample, step


def main(args):
    ipd = IPD(args.len_rollout, args.batch_size)
    agent1, agent2 = Agent(args), Agent(args)
    agent1_copy, agent2_copy = Agent(args), Agent(args)
    n_lookaheads = args.n_lookaheads
    joint_scores = []
    print('start iterations with', n_lookaheads, 'lookaheads:')

    for update in range(args.n_update):
        # reset virtual update
        agent1.set_virtual()
        agent2.set_virtual()

        # agent 2 assumes that agent 1 conducts n-step lookahead
        for _ in range(n_lookaheads):
            memory1, memory2 = sample(
                ipd,
                [agent1.virtual_theta.theta, agent2.theta],
                [agent1.values, agent2.values],
                args,
            )
            inner_loss = memory1.dice_objective(use_baseline=args.use_baseline)
            agent1.virtual_optimizer.step(inner_loss)

        # agent 1 assumes that agent 2 conducts n-step lookahead
        for _ in range(n_lookaheads):
            memory1, memory2 = sample(
                ipd,
                [agent1.theta, agent2.virtual_theta.theta],
                [agent1.values, agent2.values],
                args,
            )
            inner_loss = memory2.dice_objective(use_baseline=args.use_baseline)
            agent2.virtual_optimizer.step(inner_loss)

        # update agent 1
        memory1, memory2 = sample(
            ipd,
            [agent1.theta, agent2.virtual_theta.theta],
            [agent1.values, agent2.values],
            args,
        )
        outer_loss = memory1.dice_objective(use_baseline=args.use_baseline)
        agent1.theta_optimizer.zero_grad()
        outer_loss.backward(retain_graph=True)
        agent1.theta_optimizer.step()

        # update agent 1 value function
        v_loss = memory1.value_loss()
        agent1.value_update(v_loss)

        # update agent 2
        memory1, memory2 = sample(
            ipd,
            [agent1.virtual_theta.theta, agent2.theta],
            [agent1.values, agent2.values],
            args,
        )
        outer_loss = memory2.dice_objective(use_baseline=args.use_baseline)
        agent2.theta_optimizer.zero_grad()
        outer_loss.backward(retain_graph=True)
        agent2.theta_optimizer.step()

        # update agent 2 value function
        v_loss = memory2.value_loss()
        agent2.value_update(v_loss)

        # evaluate progress:
        score = step(ipd, agent1.theta, agent2.theta, agent1.values, agent2.values, args)
        joint_scores.append(0.5 * (score[0] + score[1]))

        if update % 10 == 0:
            p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
            p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
            print(
                'update',
                update,
                f'score ({score[0]:.3f},{score[1]:.3f})',
                'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}'
                % (p1[0], p1[1], p1[2], p1[3], p1[4]),
                f' (agent2) = {{{p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f}, {p2[3]:.3f}, {p2[4]:.3f}}}',
            )

    return joint_scores


if __name__ == '__main__':
    args = parse_args()
    joint_score = {}
    for nla in range(3):
        args.n_lookaheads = nla
        joint_score[nla] = main(args)
    np.save('result.npy', joint_score)
