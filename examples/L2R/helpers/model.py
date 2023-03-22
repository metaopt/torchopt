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
# https://github.com/uber-research/learning-to-reweight-examples
# ==============================================================================
# Copyright (c) 2017 - 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# Models for MNIST experiments.
#

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.args = args
        self.meta_weights = torch.zeros(self.args.batch_size, requires_grad=True).to(
            self.args.device,
        )
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.model(x).squeeze(dim=-1)

    def reset_meta(self, size):
        self.meta_weights = torch.zeros(size, requires_grad=True).to(self.args.device)

    def normalise(self):
        self.meta_weights = self.meta_weights.detach()
        weights_sum = torch.sum(self.meta_weights)
        weights_sum = weights_sum + 1 if weights_sum == 0 else weights_sum
        self.meta_weights /= weights_sum

    def inner_loss(self, train_x, train_y):
        result = self.forward(train_x)

        # manually implement bce_loss to make the loss differentiable w.r.t self.meta_weights
        loss = -(
            train_y * torch.log(result + 1e-10) + (1 - train_y) * torch.log(1 - result + 1e-10)
        )
        weighted_loss = torch.sum(self.meta_weights * loss)
        return weighted_loss

    def outer_loss(self, valid_x, valid_y):
        result = self.forward(valid_x)
        loss = self.criterion(result, valid_y)
        return loss
