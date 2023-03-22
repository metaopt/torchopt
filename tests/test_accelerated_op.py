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
import torch
import torch.nn.functional as F

import helpers
import torchopt


try:
    import torchopt._C.adam_op
except ImportError:
    CXX_ACCELERATED_OP_AVAILABLE = False
else:
    CXX_ACCELERATED_OP_AVAILABLE = True


def test_accelerated_op_is_available() -> None:
    assert torchopt.accelerated_op_available('cpu')
    assert torchopt.accelerated_op_available(torch.device('cpu'))

    if CXX_ACCELERATED_OP_AVAILABLE:
        assert not torchopt.accelerated_op_available('meta')
        assert not torchopt.accelerated_op_available(torch.device('meta'))
        assert not torchopt.accelerated_op_available(['cpu', 'meta'])
        assert not torchopt.accelerated_op_available([torch.device('cpu'), torch.device('meta')])
    else:
        assert torchopt.accelerated_op_available('meta')
        assert torchopt.accelerated_op_available(torch.device('meta'))
        assert torchopt.accelerated_op_available(['cpu', 'meta'])
        assert torchopt.accelerated_op_available([torch.device('cpu'), torch.device('meta')])

    if torch.cuda.is_available():
        assert torchopt.accelerated_op_available()
        assert torchopt.accelerated_op_available('cuda')
        assert torchopt.accelerated_op_available('cuda:0')
        assert torchopt.accelerated_op_available(0)
        assert torchopt.accelerated_op_available(['cpu', 'cuda'])
        assert torchopt.accelerated_op_available(['cpu', 'cuda:0'])
        assert torchopt.accelerated_op_available(['cpu', 0])
    else:
        assert not torchopt.accelerated_op_available()
        assert not torchopt.accelerated_op_available('cuda')
        assert not torchopt.accelerated_op_available('cuda:0')
        assert not torchopt.accelerated_op_available(0)
        assert not torchopt.accelerated_op_available(['cpu', 'cuda'])
        assert not torchopt.accelerated_op_available(['cpu', 'cuda:0'])
        assert not torchopt.accelerated_op_available(['cpu', 0])


@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-2, 1e-3, 1e-4],
    inplace=[True, False],
)
def test_accelerated_op(
    dtype: torch.dtype,
    lr: float,
    inplace: bool,
) -> None:
    if dtype is torch.float32 and inplace:
        return
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(
        lr,
        use_accelerated_op=True,
    )
    optim_state = optim.init(params)

    fmodel_ref, params_ref, buffers_ref = functorch.make_functional_with_buffers(model_ref)
    optim_ref = torchopt.adam(
        lr,
        use_accelerated_op=False,
    )
    optim_state_ref = optim_ref.init(params_ref)

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fmodel(params, buffers, xs)
        pred_ref = fmodel_ref(params_ref, buffers_ref, xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        grads = torch.autograd.grad(loss_ref, params_ref, allow_unused=True)
        updates, optim_state_ref = optim_ref.update(
            grads,
            optim_state_ref,
            params=params,
            inplace=inplace,
        )
        params_ref = torchopt.apply_updates(params_ref, updates, inplace=inplace)

    helpers.assert_pytree_all_close(params, params_ref)


@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    outer_lr=[1e-2, 1e-3, 1e-4],
    inner_lr=[1e-2, 1e-3, 1e-4],
    inner_update=[2, 3, 5],
    inplace=[True, False],
)
def test_maml_accelerated_op(
    dtype: torch.dtype,
    outer_lr: float,
    inner_lr: float,
    inner_update: int,
    inplace: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    outer_optim = torchopt.adam(
        outer_lr,
        use_accelerated_op=True,
    )
    outer_optim_state = outer_optim.init(params)

    fmodel_ref, params_ref, buffers_ref = functorch.make_functional_with_buffers(model_ref)
    outer_optim_ref = torchopt.adam(
        outer_lr,
        use_accelerated_op=False,
    )
    outer_optim_state_ref = outer_optim_ref.init(params_ref)

    def maml_inner_solver(params, data, use_accelerated_op):
        # Initial functional optimizer based on TorchOpt
        x, y, f, b = data
        inner_optimizer = torchopt.adam(
            inner_lr,
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
                    inplace=False,
                )  # get updates
                params = torchopt.apply_updates(params, updates, inplace=False)
        return (params, b)

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        data = (xs, ys, fmodel, buffers)
        data_ref = (xs, ys, fmodel_ref, buffers_ref)

        params_prime, buffers_prime = maml_inner_solver(params, data, use_accelerated_op=True)
        params_prime_ref, buffers_prime_ref = maml_inner_solver(
            params_ref,
            data_ref,
            use_accelerated_op=False,
        )

        pred = fmodel(params_prime, buffers_prime, xs)
        pred_ref = fmodel_ref(params_prime_ref, buffers_prime_ref, xs)
        outer_loss = F.cross_entropy(pred, ys)
        outer_loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(outer_loss, params, allow_unused=True)
        updates, outer_optim_state = outer_optim.update(
            grads,
            outer_optim_state,
            params=params,
            inplace=inplace,
        )
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        grads = torch.autograd.grad(outer_loss_ref, params_ref, allow_unused=True)
        updates, outer_optim_state_ref = outer_optim_ref.update(
            grads,
            outer_optim_state_ref,
            params=params,
            inplace=inplace,
        )
        params_ref = torchopt.apply_updates(params_ref, updates, inplace=inplace)

        torchopt.stop_gradient(model)
        torchopt.stop_gradient(model_ref)
