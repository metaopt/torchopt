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
import torch.nn.functional as F

import torchopt


def test_gamma():
    class Rollout:
        @staticmethod
        def get():
            out = torch.empty(5, 2)
            out[:, 0] = torch.randn(5)
            out[:, 1] = 0.1 * torch.ones(5)
            label = torch.arange(0, 10)
            return out.view(10, 1), F.one_hot(label, 10)

        @staticmethod
        def rollout(trajectory, gamma):
            out = [trajectory[-1]]
            for i in reversed(range(9)):
                out.append(trajectory[i] + gamma[i] * out[-1].clone().detach_())
            out.reverse()
            return torch.hstack(out).view(10, 1)

    class ValueNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    inner_iters = 1
    outer_iters = 10000
    net = ValueNetwork()
    inner_optimizer = torchopt.MetaSGD(net, lr=5e-1)
    gamma = torch.zeros(9, requires_grad=True)
    meta_optimizer = torchopt.SGD([gamma], lr=5e-1)
    net_state = torchopt.extract_state_dict(net)
    for i in range(outer_iters):
        for _ in range(inner_iters):
            trajectory, state = Rollout.get()
            backup = Rollout.rollout(trajectory, torch.sigmoid(gamma))
            pred_value = net(state.float())

            loss = F.mse_loss(pred_value, backup)
            inner_optimizer.step(loss)

        trajectory, state = Rollout.get()
        pred_value = net(state.float())
        backup = Rollout.rollout(trajectory, torch.ones_like(gamma))

        loss = F.mse_loss(pred_value, backup)
        meta_optimizer.zero_grad()
        loss.backward()
        meta_optimizer.step()
        torchopt.recover_state_dict(net, net_state)
        if i % 100 == 0:
            with torch.no_grad():
                print(f'epoch {i} | gamma: {torch.sigmoid(gamma)}')


if __name__ == '__main__':
    test_gamma()
