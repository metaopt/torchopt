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

import copy
from collections import OrderedDict
from types import FunctionType
from typing import Tuple

import functorch
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
from torch.utils import data

import helpers
import torchopt
from torchopt import pytree


BATCH_SIZE = 8
NUM_UPDATES = 5


class FcNet(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.fc = nn.Linear(in_features=dim, out_features=out, bias=True)
        nn.init.ones_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-2, 1e-3],
    method=['naive', 'forward', 'antithetic'],
    sigma=[0.01, 0.1, 1],
)
def test_zero_order(dtype: torch.dtype, lr: float, method: str, sigma: float) -> None:
    helpers.seed_everything(42)
    np_dtype = helpers.dtype_torch2numpy(dtype)
    input_size = 32
    output_size = 1
    batch_size = BATCH_SIZE
    coef = 0.1
    num_iterations = NUM_UPDATES
    num_samples = 500

    model = FcNet(input_size, output_size)

    fmodel, params = functorch.make_functional(model)
    x = torch.randn(batch_size, input_size) * coef
    y = torch.randn(input_size) * coef
    distribution = torch.distributions.Normal(loc=0, scale=1)

    @torchopt.diff.zero_order.zero_order(
        distribution=distribution, method=method, argnums=0, sigma=sigma, num_samples=num_samples
    )
    def forward_process(params, fn, x, y):
        y_pred = fn(params, x)
        loss = torch.mean((y - y_pred) ** 2)
        return loss

    optimizer = torchopt.adam(lr=lr)
    opt_state = optimizer.init(params)

    for i in range(num_iterations):
        opt_state = optimizer.init(params)  # init optimizer
        loss = forward_process(params, fmodel, x, y)  # compute loss

        grads = torch.autograd.grad(loss, params)  # compute gradients
        updates, opt_state = optimizer.update(grads, opt_state)  # get updates
        params = torchopt.apply_updates(params, updates)  # update network parameters
