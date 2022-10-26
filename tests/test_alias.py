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
    dtype=[torch.float64, torch.float32],
    lr=[1e-2, 1e-3, 1e-4],
    momentum=[0.0, 0.1],
    dampening=[0.0, 0.5],
    nesterov=[False, True],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False],  # TODO: test maximize after PyTorch 1.13
)
def test_sgd(
    dtype: torch.dtype,
    lr: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
) -> None:
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        pytest.skip('Nesterov momentum requires a momentum and zero dampening.')

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.sgd(
        lr,
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
        weight_decay=weight_decay,
        maximize=maximize,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.SGD(
        model_ref.parameters(),
        lr,
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
        weight_decay=weight_decay,
        maximize=maximize,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fmodel(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
)
def test_adam(
    dtype: torch.dtype,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
        amsgrad=False,
        weight_decay=weight_decay,
        maximize=maximize,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fmodel(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
)
def test_adamw(
    dtype: torch.dtype,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adamw(
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.AdamW(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
        amsgrad=False,
        weight_decay=weight_decay,
        maximize=maximize,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fmodel(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[1e-2, 1e-1],
    maximize=[False, True],
)
def test_adam_accelerated_cpu(
    dtype: torch.dtype,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=True,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
        amsgrad=False,
        weight_decay=weight_decay,
        maximize=maximize,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fmodel(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device available.')
@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
)
def test_adam_accelerated_cuda(
    dtype: torch.dtype,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
) -> None:
    device = 'cuda'
    model, model_ref, model_base, loader = helpers.get_models(device=device, dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=True,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
        amsgrad=False,
        weight_decay=weight_decay,
        maximize=maximize,
    )

    for xs, ys in loader:
        xs = xs.to(device=device, dtype=dtype)
        ys = ys.to(device=device)
        pred = fmodel(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    alpha=[0.9, 0.99],
    eps=[1e-8],
    momentum=[0.0, 0.1],
    centered=[False, True],
    weight_decay=[0.0, 1e-2],
    inplace=[True, False],
)
def test_rmsprop(
    dtype: torch.dtype,
    lr: float,
    alpha: float,
    eps: float,
    momentum: float,
    centered: bool,
    weight_decay: float,
    inplace: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.rmsprop(
        lr,
        alpha=alpha,
        eps=eps,
        momentum=momentum,
        centered=centered,
        nesterov=False,
        weight_decay=weight_decay,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.RMSprop(
        model_ref.parameters(),
        lr,
        alpha=alpha,
        eps=eps,
        momentum=momentum,
        centered=centered,
        weight_decay=weight_decay,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fmodel(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)
