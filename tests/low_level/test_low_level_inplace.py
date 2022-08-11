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

from typing import Tuple

import functorch
import pytest
import torch
import torch.nn.functional as F

import helpers
import torchopt


@helpers.parametrize(
    dtype=[torch.float32, torch.float64],
    lr=[1e-3, 1e-4],
    momentum=[0.0, 0.1],
    nesterov=[False, True],
)
def test_sgd(dtype: torch.dtype, lr: float, momentum: float, nesterov: bool) -> None:
    if nesterov and momentum <= 0.0:
        pytest.skip('Nesterov momentum requires a momentum and zero dampening.')

    model, model_ref, loader = helpers.get_models(device='cpu', dtype=dtype)

    fun, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.sgd(lr, momentum=(momentum if momentum != 0.0 else None), nesterov=nesterov)
    optim_state = optim.init(params)
    optim_ref = torch.optim.SGD(
        model_ref.parameters(),
        lr,
        momentum=momentum,
        dampening=0.0,
        nesterov=nesterov,
        weight_decay=0.0,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fun(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grad = torch.autograd.grad(loss, params)
        updates, optim_state = optim.update(grad, optim_state)
        params = torchopt.apply_updates(params, updates)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    with torch.no_grad():
        for p, p_ref in zip(params, model_ref.parameters()):
            helpers.assert_all_close(p, p_ref)
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            helpers.assert_all_close(b, b_ref)


@helpers.parametrize(
    dtype=[torch.float32, torch.float64],
    lr=[1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
)
def test_adam(dtype: torch.dtype, lr: float, betas: Tuple[float, float], eps: float) -> None:
    model, model_ref, loader = helpers.get_models(device='cpu', dtype=dtype)

    fun, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(lr, b1=betas[0], b2=betas[1], eps=eps, eps_root=0.0)
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(
        model_ref.parameters(), lr, betas=betas, eps=eps, amsgrad=False, weight_decay=0.0
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fun(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grad = torch.autograd.grad(loss, params)
        updates, optim_state = optim.update(grad, optim_state)
        params = torchopt.apply_updates(params, updates)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    with torch.no_grad():
        for p, p_ref in zip(params, model_ref.parameters()):
            helpers.assert_all_close(p, p_ref)
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            helpers.assert_all_close(b, b_ref)


@helpers.parametrize(
    dtype=[torch.float32, torch.float64],
    lr=[1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
)
def test_accelerated_adam_cpu(
    dtype: torch.dtype, lr: float, betas: Tuple[float, float], eps: float
) -> None:
    model, model_ref, loader = helpers.get_models(device='cpu', dtype=dtype)

    fun, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(
        lr, b1=betas[0], b2=betas[1], eps=eps, eps_root=0.0, use_accelerated_op=True
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(
        model_ref.parameters(), lr, betas=betas, eps=eps, amsgrad=False, weight_decay=0.0
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fun(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grad = torch.autograd.grad(loss, params)
        updates, optim_state = optim.update(grad, optim_state)
        params = torchopt.apply_updates(params, updates)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    with torch.no_grad():
        for p, p_ref in zip(params, model_ref.parameters()):
            helpers.assert_all_close(p, p_ref)
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            helpers.assert_all_close(b, b_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device available.')
@helpers.parametrize(
    dtype=[torch.float32, torch.float64],
    lr=[1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
)
def test_accelerated_adam_cuda(
    dtype: torch.dtype, lr: float, betas: Tuple[float, float], eps: float
) -> None:
    device = 'cuda'
    model, model_ref, loader = helpers.get_models(device=device, dtype=dtype)

    fun, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(
        lr, b1=betas[0], b2=betas[1], eps=eps, eps_root=0.0, use_accelerated_op=True
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(
        model_ref.parameters(), lr, betas=betas, eps=eps, amsgrad=False, weight_decay=0.0
    )

    for xs, ys in loader:
        xs = xs.to(device=device, dtype=dtype)
        ys = ys.to(device=device)
        pred = fun(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grad = torch.autograd.grad(loss, params)
        updates, optim_state = optim.update(grad, optim_state)
        params = torchopt.apply_updates(params, updates)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    with torch.no_grad():
        for p, p_ref in zip(params, model_ref.parameters()):
            helpers.assert_all_close(p, p_ref)
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            helpers.assert_all_close(b, b_ref)


@helpers.parametrize(
    dtype=[torch.float32, torch.float64],
    lr=[1e-3, 1e-4],
    alpha=[0.9, 0.99],
    eps=[1e-8],
    momentum=[0.0, 0.1],
    centered=[False, True],
)
def test_rmsprop(
    dtype: torch.dtype, lr: float, alpha: float, eps: float, momentum: float, centered: bool
) -> None:
    model, model_ref, loader = helpers.get_models(device='cpu', dtype=dtype)

    fun, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.rmsprop(
        lr,
        decay=alpha,
        eps=eps,
        momentum=(momentum if momentum != 0.0 else None),
        centered=centered,
        nesterov=False,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.RMSprop(
        model_ref.parameters(),
        lr,
        alpha=alpha,
        eps=eps,
        momentum=momentum,
        centered=centered,
        weight_decay=0.0,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fun(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grad = torch.autograd.grad(loss, params)
        updates, optim_state = optim.update(grad, optim_state)
        params = torchopt.apply_updates(params, updates)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    with torch.no_grad():
        for p, p_ref in zip(params, model_ref.parameters()):
            helpers.assert_all_close(p, p_ref)
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            helpers.assert_all_close(b, b_ref)
