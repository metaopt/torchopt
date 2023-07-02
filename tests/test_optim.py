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

from __future__ import annotations

from typing import Callable

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
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
)
def test_SGD(
    dtype: torch.dtype,
    lr: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    weight_decay: float,
    maximize: bool,
) -> None:
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        pytest.skip('Nesterov momentum requires a momentum and zero dampening.')

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    optim = torchopt.SGD(
        model.parameters(),
        lr,
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
        weight_decay=weight_decay,
        maximize=maximize,
    )
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
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_accelerated_op=[False, True],
)
def test_Adam(
    dtype: torch.dtype,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool,
    use_accelerated_op: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    optim = torchopt.Adam(
        model.parameters(),
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=use_accelerated_op,
    )
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
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    rho=[0.9, 0.95],
    eps=[1e-8],
    weight_decay=[0.0, 1e-2],
)
def test_Adadelta(
    dtype: torch.dtype,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    optim = torchopt.Adadelta(
        model.parameters(),
        lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
    )
    optim_ref = torch.optim.Adadelta(
        model_ref.parameters(),
        lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    weight_decay=[0.0, 1e-2],
)
def test_RAdam(
    dtype: torch.dtype,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    optim = torchopt.RAdam(
        model.parameters(),
        lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    optim_ref = torch.optim.RAdam(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    weight_decay=[0.0, 1e-2],
)
def test_Adamax(
    dtype: torch.dtype,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    optim = torchopt.Adamax(
        model.parameters(),
        lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    optim_ref = torch.optim.Adamax(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    weight_decay=[1e-2, 1e-1],
    maximize=[False, True],
    use_accelerated_op=[False, True],
)
def test_AdamW(
    dtype: torch.dtype,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool,
    use_accelerated_op: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    optim = torchopt.AdamW(
        model.parameters(),
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=use_accelerated_op,
    )
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
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device available.')
@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    optimizers=[
        (torchopt.Adam, torch.optim.Adam),
        (torchopt.AdamW, torch.optim.AdamW),
    ],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
)
def test_Adam_accelerated_cuda(
    dtype: torch.dtype,
    lr: float,
    optimizers: tuple[torchopt.Optimizer, torch.optim.Optimizer],
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    device = 'cuda'
    model, model_ref, model_base, loader = helpers.get_models(device=device, dtype=dtype)

    torchopt_optimizer, torch_optimizer = optimizers

    optim = torchopt_optimizer(
        model.parameters(),
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=True,
    )
    optim_ref = torch_optimizer(
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
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    lr_decay=[0.0, 1e-2],
    initial_accumulator_value=[0.0, 1e-1],
    eps=[1e-8],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
)
def test_AdaGrad(
    dtype: torch.dtype,
    lr: float,
    lr_decay: float,
    initial_accumulator_value: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    optim = torchopt.AdaGrad(
        model.parameters(),
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        initial_accumulator_value=initial_accumulator_value,
        eps=eps,
        maximize=maximize,
    )
    optim_ref = torch.optim.Adagrad(
        model_ref.parameters(),
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        initial_accumulator_value=initial_accumulator_value,
        eps=eps,
        maximize=maximize,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    alpha=[0.9, 0.99],
    eps=[1e-8],
    momentum=[0.0, 0.1],
    centered=[False, True],
    weight_decay=[0.0, 1e-2],
)
def test_RMSProp(
    dtype: torch.dtype,
    lr: float,
    alpha: float,
    eps: float,
    momentum: float,
    centered: bool,
    weight_decay: float,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    optim = torchopt.RMSProp(
        model.parameters(),
        lr,
        alpha=alpha,
        eps=eps,
        momentum=momentum,
        centered=centered,
        nesterov=False,
        weight_decay=weight_decay,
    )
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
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)


@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-2, 1e-3],
    optimizers=[
        (torchopt.sgd, torch.optim.SGD, {}),
        (torchopt.adam, torch.optim.Adam, {}),
        (torchopt.adamw, torch.optim.AdamW, {}),
        (torchopt.adagrad, torch.optim.Adagrad, {'eps': 1e-8}),
        (torchopt.rmsprop, torch.optim.RMSprop, {}),
    ],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
)
def test_FuncOptimizer(
    dtype: torch.dtype,
    lr: float,
    optimizers: tuple[Callable, torch.optim.Optimizer],
    inplace: bool,
    weight_decay: float,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    torchopt_optimizer, torch_optimizer, optimizer_kwargs = optimizers

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.FuncOptimizer(
        torchopt_optimizer(
            lr=lr,
            weight_decay=weight_decay,
            **optimizer_kwargs,
        ),
        inplace=inplace,
    )
    optim_ref = torch_optimizer(
        model_ref.parameters(),
        lr,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fmodel(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        params = optim.step(loss, params)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)
