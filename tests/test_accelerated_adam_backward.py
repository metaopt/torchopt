# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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

import importlib
import math

import pytest
import torch

import torchopt
from torchopt.accelerated_op import AdamOp
from torchopt.accelerated_op._src import adam_op as python_adam_op


@pytest.fixture(
    params=[('python', 'cpu'), ('native', 'cpu'), ('python', 'cuda'), ('native', 'cuda')],
)
def adam_backend(request, monkeypatch):
    backend, device = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    if backend == 'native':
        implementation = pytest.importorskip('torchopt._C.adam_op')
        if device == 'cuda':
            probe = torch.ones(1, device=device)
            try:
                implementation.forward_mu(probe, probe, 0.9)
            except RuntimeError as error:
                if str(error) != 'Not implemented':
                    raise
                pytest.skip('The native extension was built without CUDA')
    else:
        implementation = python_adam_op
    module = importlib.import_module('torchopt.accelerated_op.adam_op')
    monkeypatch.setattr(module, 'adam_op', implementation)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield device
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize(('eps', 'eps_root'), [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.01, 0.01)])
@pytest.mark.parametrize('count', [1, 7])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_adam_update_adjoint(adam_backend, eps, eps_root, count, dtype):
    mu = torch.tensor([0.0, 0.2, -0.3], dtype=dtype, device=adam_backend, requires_grad=True)
    nu = torch.tensor([0.03, 0.005, 0.2], dtype=dtype, device=adam_backend, requires_grad=True)
    cotangent = torch.tensor([-0.7, 0.1, 1.2], dtype=dtype, device=adam_backend)
    saved = [tensor.detach().clone() for tensor in (mu, nu, cotangent)]
    b1, b2 = 0.8, 0.95

    actual = AdamOp.UpdatesOp.apply(mu, nu, (b1, b2, eps, eps_root, count))
    expected = (mu / (1 - b1**count)) / ((nu / (1 - b2**count) + eps_root).sqrt() + eps)
    actual_grads = torch.autograd.grad(actual, (mu, nu), cotangent)
    expected_grads = torch.autograd.grad(expected, (mu, nu), cotangent)

    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    for tensor, original in zip((mu, nu, cotangent), saved):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


@pytest.mark.parametrize(('eps', 'eps_root'), [(0.1, 0.0), (0.0, 0.01), (0.01, 0.01)])
@pytest.mark.parametrize('count', [1, 5])
def test_adam_update_gradcheck(adam_backend, eps, eps_root, count):
    mu = torch.tensor(
        [0.0, 0.2, -0.3],
        dtype=torch.float64,
        device=adam_backend,
        requires_grad=True,
    )
    nu = torch.tensor(
        [0.03, 0.005, 0.2],
        dtype=torch.float64,
        device=adam_backend,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(
        lambda m, v: AdamOp.UpdatesOp.apply(m, v, (0.9, 0.999, eps, eps_root, count)),
        (mu, nu),
    )


@pytest.mark.parametrize('initial', [0.0, 0.3, -0.3])
@pytest.mark.parametrize(('eps', 'eps_root'), [(1e-4, 1e-3), (0.1, 0.0), (0.0, 0.1)])
def test_adam_one_step_meta_gradient(adam_backend, initial, eps, eps_root):
    x = torch.tensor(initial, dtype=torch.float64, device=adam_backend, requires_grad=True)
    optimizer = torchopt.adam(lr=0.1, eps=eps, eps_root=eps_root, use_accelerated_op=True)
    state = optimizer.init([x])
    gradient = torch.autograd.grad(x.square() / 2, x, create_graph=True)[0]
    updates, _ = optimizer.update([gradient], state, inplace=False)
    output = torchopt.apply_updates([x], updates, inplace=False)[0]
    meta_gradient = torch.autograd.grad(output, x)[0]

    root = math.sqrt(initial**2 + eps_root)
    denominator = root + eps
    expected_gradient = 1 - 0.1 / denominator
    if root > 0:
        expected_gradient += 0.1 * initial**2 / (root * denominator**2)
    torch.testing.assert_close(output, x - 0.1 * x / denominator)
    torch.testing.assert_close(meta_gradient, meta_gradient.new_tensor(expected_gradient))

    # This also checks the unsmoothed zero state using the public forward map.
    def forward(value):
        return value - 0.1 * value / (math.sqrt(value**2 + eps_root) + eps)

    step = 1e-7
    numerical_gradient = (forward(initial + step) - forward(initial - step)) / (2 * step)
    assert meta_gradient.item() == pytest.approx(numerical_gradient, abs=2e-6)


@pytest.mark.parametrize('steps', [2, 5])
@pytest.mark.parametrize(('eps', 'eps_root'), [(0.1, 0.0), (0.0, 0.01), (0.01, 0.01)])
def test_adam_multi_step_meta_gradient(adam_backend, steps, eps, eps_root):
    initial = torch.tensor([0.0, 0.3, -0.2], dtype=torch.float64, device=adam_backend)

    def solve(value, accelerated):
        x = value.detach().clone().requires_grad_()
        params = [x]
        optimizer = torchopt.adam(
            lr=0.05,
            eps=eps,
            eps_root=eps_root,
            use_accelerated_op=accelerated,
        )
        state = optimizer.init(params)
        weights = x.new_tensor([0.7, 1.3, 0.4])
        for index in range(steps):
            target = x.new_tensor([0.1, -0.2, 0.3]) * index / (index + 1)
            loss = ((params[0] - target).square() * weights).sum() / 2
            gradients = list(torch.autograd.grad(loss, params, create_graph=True))
            updates, state = optimizer.update(gradients, state, inplace=False)
            params = torchopt.apply_updates(params, updates, inplace=False)
        objective = (params[0] * x.new_tensor([0.2, -0.5, 0.8])).sum()
        return objective, torch.autograd.grad(objective, x)[0]

    actual, actual_gradient = solve(initial, True)
    expected, expected_gradient = solve(initial, False)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    # Standard autograd has a sqrt(0) singularity when eps_root=0; the public
    # composite map still has a finite first derivative at a zero gradient.
    if eps_root > 0:
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-10)
    step = 1e-7
    for index in range(initial.numel()):
        offset = torch.zeros_like(initial)
        offset[index] = step
        plus, _ = solve(initial + offset, False)
        minus, _ = solve(initial - offset, False)
        numerical = (plus - minus) / (2 * step)
        torch.testing.assert_close(actual_gradient[index], numerical, rtol=1e-5, atol=2e-6)


def test_adam_zero_variance_mixed_lanes(adam_backend):
    # More than one CUDA unroll block, with a zero state before nonzero lanes.
    size = 4096
    mu = torch.full((size,), 0.2, dtype=torch.float64, device=adam_backend, requires_grad=True)
    nu = torch.full((size,), 0.3, dtype=torch.float64, device=adam_backend, requires_grad=True)
    with torch.no_grad():
        mu[::4] = 0
        nu[::4] = 0
    result = AdamOp.UpdatesOp.apply(mu, nu, (0.9, 0.99, 0.1, 0.0, 1))
    dmu, dnu = torch.autograd.grad(result, (mu, nu), torch.ones_like(result))
    root = (nu / 0.01).sqrt()
    denominator = root + 0.1
    torch.testing.assert_close(dmu, 1 / (0.1 * denominator))
    torch.testing.assert_close(dnu[::4], torch.zeros_like(dnu[::4]))
    nonzero = mu != 0
    expected_dnu = -mu[nonzero] / (2 * 0.1 * 0.01 * root[nonzero] * denominator[nonzero].square())
    torch.testing.assert_close(dnu[nonzero], expected_dnu)


def test_adam_zero_denominator_is_undefined(adam_backend):
    mu = torch.zeros(1, dtype=torch.float64, device=adam_backend)
    nu = torch.zeros_like(mu)
    result = AdamOp.UpdatesOp.apply(mu, nu, (0.9, 0.99, 0.0, 0.0, 1))
    assert torch.isnan(result).all()
