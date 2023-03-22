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
import torchviz

import torchopt


class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x, meta_param):
        return self.fc(x) + meta_param


def draw_torchviz():
    net = Net(dim).cuda()
    optimizer = torchopt.MetaAdam(net, lr=1e-3, use_accelerated_op=False)
    meta_param = torch.tensor(1.0, requires_grad=True)

    xs = torch.ones(batch_size, dim).cuda()

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))
    optimizer.step(loss)

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))
    # draw computation graph
    torchviz.make_dot(loss).render('torchviz_graph', format='svg')


def draw_torchopt():
    net = Net(dim).cuda()
    optimizer = torchopt.MetaAdam(net, lr=1e-3, use_accelerated_op=True)
    meta_param = torch.tensor(1.0, requires_grad=True)

    xs = torch.ones(batch_size, dim).cuda()

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))
    # set enable_visual
    net_state_0 = torchopt.extract_state_dict(net, enable_visual=True, visual_prefix='step0.')
    optimizer.step(loss)
    # set enable_visual
    net_state_1 = torchopt.extract_state_dict(net, enable_visual=True, visual_prefix='step1.')

    pred = net(xs, meta_param)
    loss = F.mse_loss(pred, torch.ones_like(pred))
    # draw computation graph
    torchopt.visual.make_dot(loss, [net_state_0, net_state_1, {meta_param: 'meta_param'}]).render(
        'torchopt_graph',
        format='svg',
    )


if __name__ == '__main__':
    dim = 5
    batch_size = 2
    draw_torchviz()
    draw_torchopt()
