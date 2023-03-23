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

import functorch
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types

import helpers
import torchopt


BATCH_SIZE = 8
NUM_UPDATES = 5


class FcNet(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.fc = nn.Linear(in_features=dim, out_features=out, bias=True)

    def forward(self, x):
        return self.fc(x)


@helpers.parametrize(
    lr=[1e-2, 1e-3],
    method=['naive', 'forward', 'antithetic'],
    sigma=[0.01, 0.1, 1],
)
def test_zero_order(lr: float, method: str, sigma: float) -> None:
    helpers.seed_everything(42)
    input_size = 32
    output_size = 1
    batch_size = BATCH_SIZE
    coef = 0.1
    num_iterations = NUM_UPDATES
    num_samples = 500

    model = FcNet(input_size, output_size)

    fmodel, params = functorch.make_functional(model)
    x = torch.randn(batch_size, input_size) * coef
    y = torch.randn(batch_size, 1) * coef
    distribution = torch.distributions.Normal(loc=0, scale=1)

    @torchopt.diff.zero_order(
        distribution=distribution,
        method=method,
        argnums=0,
        sigma=sigma,
        num_samples=num_samples,
    )
    def forward_process(params, fn, x, y):
        y_pred = fn(params, x)
        return F.mse_loss(y_pred, y)

    optimizer = torchopt.adam(lr=lr)
    opt_state = optimizer.init(params)  # init optimizer

    for _ in range(num_iterations):
        loss = forward_process(params, fmodel, x, y)  # compute loss

        grads = torch.autograd.grad(loss, params)  # compute gradients
        updates, opt_state = optimizer.update(grads, opt_state)  # get updates
        params = torchopt.apply_updates(params, updates)  # update network parameters


@helpers.parametrize(
    lr=[1e-2, 1e-3],
    method=['naive', 'forward', 'antithetic'],
    sigma=[0.01, 0.1, 1],
)
def test_zero_order_module(lr: float, method: str, sigma: float) -> None:
    helpers.seed_everything(42)
    input_size = 32
    output_size = 1
    batch_size = BATCH_SIZE
    coef = 0.1
    num_iterations = NUM_UPDATES
    num_samples = 500

    class FcNetWithLoss(
        torchopt.nn.ZeroOrderGradientModule,
        method=method,
        sigma=sigma,
        num_samples=num_samples,
    ):
        def __init__(self, dim, out):
            super().__init__()
            self.net = FcNet(dim, out)
            self.loss = nn.MSELoss()
            self.distribution = torch.distributions.Normal(loc=0, scale=1)

        def forward(self, x, y):
            return self.loss(self.net(x), y)

        def sample(self, sample_shape=torch.Size()):  # noqa: B008
            return self.distribution.sample(sample_shape)

    x = torch.randn(batch_size, input_size) * coef
    y = torch.randn(batch_size, 1) * coef
    model_with_loss = FcNetWithLoss(input_size, output_size)

    optimizer = torchopt.Adam(model_with_loss.parameters(), lr=lr)

    for _ in range(num_iterations):
        loss = model_with_loss(x, y)  # compute loss

        optimizer.zero_grad()
        loss.backward()  # compute gradients
        optimizer.step()  # update network parameters


def test_module_enable_zero_order_gradients_twice() -> None:
    class MyModule(torchopt.nn.ZeroOrderGradientModule):
        def forward(self):
            return torch.tensor(0.0)

        def sample(self, sample_shape):
            return torch.tensor(0.0)

    from torchopt.diff.zero_order.nn.module import enable_zero_order_gradients

    with pytest.raises(
        TypeError,
        match='Zero-order gradient estimation is already enabled for the `forward` method.',
    ):
        enable_zero_order_gradients(MyModule)


def test_module_empty_parameters() -> None:
    class MyModule(torchopt.nn.ZeroOrderGradientModule):
        def forward(self):
            return torch.tensor(0.0)

        def sample(self, sample_shape):
            return torch.tensor(0.0)

    m = MyModule()
    with pytest.raises(RuntimeError, match='The module has no parameters.'):
        m()


def test_module_abstract_methods() -> None:
    class MyModule1(torchopt.nn.ZeroOrderGradientModule):
        def forward(self):
            return torch.tensor(0.0)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MyModule1()

    class MyModule2(torchopt.nn.ZeroOrderGradientModule):
        def sample(self, sample_shape):
            return torch.tensor(0.0)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MyModule2()
