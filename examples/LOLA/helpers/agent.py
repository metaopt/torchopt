# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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

import torch
import torch.nn as nn

import torchopt


class theta_model(nn.Module):
    def __init__(self, theta):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(theta.detach(), requires_grad=True))


class Agent:
    def __init__(self, args):
        self.args = args
        # init theta and its optimizer
        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.theta_optimizer = torch.optim.Adam((self.theta,), lr=args.lr_out)

        # init values and its optimizer
        self.values = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.value_optimizer = torch.optim.Adam((self.values,), lr=args.lr_v)

        self.set_virtual()

    def set_virtual(self):
        self.virtual_theta = theta_model(self.theta)
        self.virtual_optimizer = torchopt.MetaSGD(self.virtual_theta, lr=self.args.lr_in)

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
