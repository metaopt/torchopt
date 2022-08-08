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
import itertools
import random
from typing import Optional, Tuple, Union

import functorch
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import models

import torchopt


def get_models(
    device: Optional[Union[str, torch.device]] = None, dtype: torch.dtype = torch.float32
) -> Tuple[nn.Module, nn.Module, data.DataLoader]:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = models.resnet18().to(dtype=dtype)
    model_ref = copy.deepcopy(model)
    if device is not None:
        model = model.to(device=torch.device(device))
        model_ref = model_ref.to(device=torch.device(device))

    batch_size = 8
    dataset = data.TensorDataset(
        torch.randn(batch_size * 2, 3, 224, 224), torch.randint(0, 1000, (batch_size * 2,))
    )
    loader = data.DataLoader(dataset, batch_size, shuffle=False)

    return model, model_ref, loader


@pytest.mark.parametrize(
    ('dtype', 'lr', 'momentum', 'nesterov'),
    list(
        itertools.product(
            [torch.float32, torch.float64],
            [1e-3, 1e-4, 1e-5],
            [0.0, 0.1, 0.2],
            [False, True],
        )
    ),
)  # fmt: skip
def test_sgd(dtype: torch.dtype, lr: float, momentum: float, nesterov: bool) -> None:
    model, model_ref, loader = get_models(device='cpu', dtype=dtype)

    fun, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.sgd(lr, momentum=(momentum if momentum != 0.0 else None), nesterov=nesterov)
    optim_state = optim.init(params)
    optim_ref = torch.optim.SGD(
        model_ref.parameters(), lr, momentum=momentum, nesterov=nesterov, weight_decay=0.0
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
            assert torch.allclose(p, p_ref), f'{p!r} != {p_ref!r}'
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            assert torch.allclose(b, b_ref), f'{b!r} != {b_ref!r}'


@pytest.mark.parametrize(
    ('dtype', 'lr', 'betas', 'eps'),
    list(
        itertools.product(
            [torch.float32, torch.float64],
            [1e-3, 1e-4, 1e-5],
            [(0.9, 0.999), (0.95, 0.9995)],
            [1e-8, 1e-6],
        )
    ),
)  # fmt: skip
def test_adam(dtype: torch.dtype, lr: float, betas: Tuple[float, float], eps: float) -> None:
    model, model_ref, loader = get_models(device='cpu', dtype=dtype)

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
            assert torch.allclose(p, p_ref), f'{p!r} != {p_ref!r}'
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            assert torch.allclose(b, b_ref), f'{b!r} != {b_ref!r}'


@pytest.mark.parametrize(
    ('dtype', 'lr', 'betas', 'eps'),
    list(
        itertools.product(
            [torch.float32, torch.float64],
            [1e-3, 1e-4, 1e-5],
            [(0.9, 0.999), (0.95, 0.9995)],
            [1e-8, 1e-6],
        )
    ),
)  # fmt: skip
def test_accelerated_adam_cpu(
    dtype: torch.dtype, lr: float, betas: Tuple[float, float], eps: float
) -> None:
    model, model_ref, loader = get_models(device='cpu', dtype=dtype)

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
            assert torch.allclose(p, p_ref), f'{p!r} != {p_ref!r}'
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            assert torch.allclose(b, b_ref), f'{b!r} != {b_ref!r}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device available.')
@pytest.mark.parametrize(
    ('dtype', 'lr', 'betas', 'eps'),
    list(
        itertools.product(
            [torch.float32, torch.float64],
            [1e-3, 1e-4, 1e-5],
            [(0.9, 0.999), (0.95, 0.9995)],
            [1e-8, 1e-6],
        )
    ),
)  # fmt: skip
def test_accelerated_adam_cuda(
    dtype: torch.dtype, lr: float, betas: Tuple[float, float], eps: float
) -> None:
    device = 'cuda'
    model, model_ref, loader = get_models(device=device, dtype=dtype)

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
            assert torch.allclose(p, p_ref), f'{p!r} != {p_ref!r}'
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            assert torch.allclose(b, b_ref), f'{b!r} != {b_ref!r}'


@pytest.mark.parametrize(
    ('dtype', 'lr', 'alpha', 'eps', 'momentum', 'centered'),
    list(
        itertools.product(
            [torch.float32, torch.float64],
            [1e-3, 1e-4, 1e-5],
            [0.9, 0.99],
            [1e-8, 1e-6],
            [0.0, 0.1, 0.2],
            [False, True],
        )
    ),
)  # fmt: skip
def test_rmsprop(
    dtype: torch.dtype, lr: float, alpha: float, eps: float, momentum: float, centered: bool
) -> None:
    model, model_ref, loader = get_models(device='cpu', dtype=dtype)

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
            assert torch.allclose(p, p_ref), f'{p!r} != {p_ref!r}'
        for b, b_ref in zip(buffers, model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            assert torch.allclose(b, b_ref), f'{b!r} != {b_ref!r}'
