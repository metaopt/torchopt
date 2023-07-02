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
from torchopt import pytree
from torchopt.alias.utils import _set_use_chain_flat
from torchopt.typing import TensorTree


@helpers.parametrize(
    optimizer=[
        torchopt.sgd,
        torchopt.adam,
        torchopt.adamw,
        torchopt.rmsprop,
    ],
    tensortree=[
        {},
        (),
        [],
        (None,),
        {'a': (), 'b': {'c': []}, 'd': None},
    ],
    maximize=[False, True],
    inplace=[True, False],
    use_chain_flat=[True, False],
)
def test_empty(
    optimizer: Callable,
    tensortree: TensorTree,
    maximize: bool,
    inplace: bool,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    params = pytree.tree_map(lambda x: x, tensortree)
    grads = pytree.tree_map(lambda x: x, tensortree)

    optim = optimizer(1e-3, maximize=maximize)
    optim_state = optim.init(params)
    updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
    _ = torchopt.apply_updates(params, updates)

    try:
        optim = optimizer(1e-3, maximize=maximize, use_accelerated_op=True)
    except TypeError:
        pass
    else:
        optim_state = optim.init(params)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        _ = torchopt.apply_updates(params, updates)

    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-2, 1e-3, 1e-4],
    momentum=[0.0, 0.1],
    dampening=[0.0, 0.5],
    nesterov=[False, True],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_chain_flat=[True, False],
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
    use_chain_flat: bool,
) -> None:
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        pytest.skip('Nesterov momentum requires a momentum and zero dampening.')

    _set_use_chain_flat(use_chain_flat)

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
    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    rho=[0.9, 0.95],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    use_chain_flat=[True, False],
)
def test_adadelta(
    dtype: torch.dtype,
    lr: float,
    rho: float,
    eps: float,
    inplace: bool,
    weight_decay: float,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adadelta(
        lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adadelta(
        model_ref.parameters(),
        lr,
        rho=rho,
        eps=eps,
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
    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_accelerated_op=[False, True],
    use_chain_flat=[True, False],
)
def test_adam(
    dtype: torch.dtype,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
    use_accelerated_op: bool,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=use_accelerated_op,
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
    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    use_chain_flat=[True, False],
)
def test_radam(
    dtype: torch.dtype,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.radam(
        lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.RAdam(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
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
    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    use_chain_flat=[True, False],
)
def test_adamax(
    dtype: torch.dtype,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adamax(
        lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adamax(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
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
    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64],
    outer_lr=[1e-2, 1e-3, 1e-4],
    inner_lr=[1e-2, 1e-3, 1e-4],
    inner_update=[2, 3, 5],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_accelerated_op=[False, True],
    use_chain_flat=[True, False],
)
def test_maml_adam(
    dtype: torch.dtype,
    outer_lr: float,
    inner_lr: float,
    inner_update: int,
    betas: tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
    use_accelerated_op: bool,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    outer_optim = torchopt.adam(
        outer_lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=use_accelerated_op,
    )
    outer_optim_state = outer_optim.init(params)

    def maml_inner_solver_torchopt(params, data, use_accelerated_op):
        # Initial functional optimizer based on TorchOpt
        x, y, f, b = data
        inner_optimizer = torchopt.adam(
            inner_lr,
            betas=betas,
            eps=eps,
            eps_root=0.0,
            weight_decay=weight_decay,
            maximize=maximize,
            use_accelerated_op=use_accelerated_op,
        )
        inner_opt_state = inner_optimizer.init(params)
        with torch.enable_grad():
            # Temporarily enable gradient computation for conducting the optimization
            for _ in range(inner_update):
                pred = f(params, b, x)
                inner_loss = F.cross_entropy(pred, y)  # compute loss
                grads = torch.autograd.grad(
                    inner_loss,
                    params,
                    allow_unused=True,
                )  # compute gradients
                updates, inner_opt_state = inner_optimizer.update(
                    grads,
                    inner_opt_state,
                    params=params,
                    inplace=False,
                )  # get updates
                params = torchopt.apply_updates(params, updates, inplace=False)
        return (params, b)

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        data = (xs, ys, fmodel, buffers)

        params_prime, buffers_prime = maml_inner_solver_torchopt(
            params,
            data,
            use_accelerated_op=True,
        )
        pred = fmodel(params_prime, buffers_prime, xs)
        outer_loss = F.cross_entropy(pred, ys)

        grads = torch.autograd.grad(outer_loss, params, allow_unused=True)
        updates, outer_optim_state = outer_optim.update(
            grads,
            outer_optim_state,
            params=params,
            inplace=inplace,
        )
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        torchopt.stop_gradient(model)

    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_accelerated_op=[False, True],
    use_chain_flat=[True, False],
)
def test_adamw(
    dtype: torch.dtype,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
    use_accelerated_op: bool,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adamw(
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=use_accelerated_op,
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
    _set_use_chain_flat(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No CUDA device available.')
@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    optimizers=[
        (torchopt.adam, torch.optim.Adam),
        (torchopt.adamw, torch.optim.AdamW),
    ],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_chain_flat=[True, False],
)
def test_adam_accelerated_cuda(
    dtype: torch.dtype,
    lr: float,
    optimizers: tuple[Callable, torch.optim.Optimizer],
    betas: tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    device = 'cuda'
    model, model_ref, model_base, loader = helpers.get_models(device=device, dtype=dtype)

    torchopt_optimizer, torch_optimizer = optimizers

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt_optimizer(
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=True,
    )
    optim_state = optim.init(params)
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
    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    lr_decay=[0.0, 1e-2],
    initial_accumulator_value=[0.0, 1e-1],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_chain_flat=[True, False],
)
def test_adagrad(
    dtype: torch.dtype,
    lr: float,
    lr_decay: float,
    initial_accumulator_value: float,
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adagrad(
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        initial_accumulator_value=initial_accumulator_value,
        eps=eps,
        maximize=maximize,
    )
    optim_state = optim.init(params)
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
    _set_use_chain_flat(True)


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    alpha=[0.9, 0.99],
    eps=[1e-8],
    momentum=[0.0, 0.1],
    centered=[False, True],
    weight_decay=[0.0, 1e-2],
    inplace=[True, False],
    use_chain_flat=[True, False],
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
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

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
    _set_use_chain_flat(True)
