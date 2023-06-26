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

import copy
import re
from collections import OrderedDict
from types import FunctionType

import functorch
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types
from torch.utils import data

import helpers
import torchopt
from torchopt import pytree
from torchopt.diff.implicit import ImplicitMetaGradientModule


try:
    import jax
    import jax.numpy as jnp
    import jaxopt
    import optax

    HAS_JAX = True
except ImportError:
    jax = jnp = jaxopt = optax = None
    HAS_JAX = False


BATCH_SIZE = 8
NUM_UPDATES = 3

MODEL_NUM_INPUTS = 10
MODEL_NUM_CLASSES = 10


class FcNet(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.fc = nn.Linear(in_features=dim, out_features=out, bias=True)
        nn.init.ones_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


def get_model_jax(dtype: np.dtype = np.float32) -> tuple[FunctionType, OrderedDict]:
    helpers.seed_everything(seed=42)

    def func(params, x):
        return x @ params['weight'] + params['bias']

    params = OrderedDict(
        [
            ('weight', jnp.ones((MODEL_NUM_INPUTS, MODEL_NUM_CLASSES), dtype=dtype)),
            ('bias', jnp.zeros((MODEL_NUM_CLASSES,), dtype=dtype)),
        ],
    )
    return func, params


@torch.no_grad()
def get_model_torch(
    device: torch.types.Device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, data.DataLoader]:
    helpers.seed_everything(seed=42)

    model = FcNet(MODEL_NUM_INPUTS, MODEL_NUM_CLASSES).to(dtype=dtype)

    if device is not None:
        model = model.to(device=torch.device(device))

    dataset = data.TensorDataset(
        torch.randint(0, 1, (BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS)),
        torch.randint(0, MODEL_NUM_CLASSES, (BATCH_SIZE * NUM_UPDATES,)),
    )
    loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=False)

    return model, loader


@torch.no_grad()
def get_rr_dataset_torch() -> data.DataLoader:
    helpers.seed_everything(seed=42)

    BATCH_SIZE = 1024
    NUM_UPDATES = 4
    dataset = data.TensorDataset(
        torch.randn((BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS)),
        torch.randn((BATCH_SIZE * NUM_UPDATES,)),
        torch.randn((BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS)),
        torch.randn((BATCH_SIZE * NUM_UPDATES,)),
    )
    return data.DataLoader(dataset, BATCH_SIZE, shuffle=False)


@pytest.mark.skipif(not HAS_JAX, reason='JAX is not installed')
@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-3, 1e-4],
    inner_lr=[2e-2, 2e-3],
    inner_update=[20, 50, 100],
)
def test_imaml_solve_normal_cg(
    dtype: torch.dtype,
    lr: float,
    inner_lr: float,
    inner_update: int,
) -> None:
    np_dtype = helpers.dtype_torch2numpy(dtype)

    jax_model, jax_params = get_model_jax(dtype=np_dtype)
    model, loader = get_model_torch(device='cpu', dtype=dtype)

    fmodel, params = functorch.make_functional(model)
    optim = torchopt.sgd(lr)
    optim_state = optim.init(params)

    optim_jax = optax.sgd(lr)
    optim_state_jax = optim_jax.init(jax_params)

    def imaml_objective_torchopt(params, meta_params, data):
        x, y, f = data
        y_pred = f(params, x)
        regularization_loss = 0
        for p1, p2 in zip(params, meta_params):
            regularization_loss += 0.5 * torch.sum(torch.square(p1 - p2))
        return F.cross_entropy(y_pred, y) + regularization_loss

    @torchopt.diff.implicit.custom_root(
        functorch.grad(imaml_objective_torchopt, argnums=0),
        argnums=1,
        has_aux=True,
        solve=torchopt.linear_solve.solve_normal_cg(),
    )
    def inner_solver_torchopt(params, meta_params, data):
        # Initial functional optimizer based on TorchOpt
        x, y, f = data
        optimizer = torchopt.sgd(lr=inner_lr)
        opt_state = optimizer.init(params)
        with torch.enable_grad():
            # Temporarily enable gradient computation for conducting the optimization
            for _ in range(inner_update):
                pred = f(params, x)
                loss = F.cross_entropy(pred, y)  # compute loss
                # Compute regularization loss
                regularization_loss = 0
                for p1, p2 in zip(params, meta_params):
                    regularization_loss += 0.5 * torch.sum(torch.square(p1 - p2))
                final_loss = loss + regularization_loss
                grads = torch.autograd.grad(final_loss, params)  # compute gradients
                updates, opt_state = optimizer.update(grads, opt_state, inplace=True)  # get updates
                params = torchopt.apply_updates(params, updates, inplace=True)
        return params, (0, {'a': 1, 'b': 2})

    def imaml_objective_jax(params, meta_params, x, y):
        y_pred = jax_model(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_pred, y))
        regularization_loss = 0
        for p1, p2 in zip(params.values(), meta_params.values()):
            regularization_loss += 0.5 * jnp.sum(jnp.square(p1 - p2))
        return loss + regularization_loss

    @jaxopt.implicit_diff.custom_root(
        jax.grad(imaml_objective_jax, argnums=0),
        has_aux=True,
        solve=jaxopt.linear_solve.solve_normal_cg,
    )
    def inner_solver_jax(params, meta_params, x, y):
        """Solve ridge regression by conjugate gradient."""
        # Initial functional optimizer based on torchopt
        optimizer = optax.sgd(inner_lr)
        opt_state = optimizer.init(params)

        def compute_loss(params, meta_params, x, y):
            pred = jax_model(params, x)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, y))
            # Compute regularization loss
            regularization_loss = 0
            for p1, p2 in zip(params.values(), meta_params.values()):
                regularization_loss += 0.5 * jnp.sum(jnp.square(p1 - p2))
            return loss + regularization_loss

        for _ in range(inner_update):
            grads = jax.grad(compute_loss)(params, meta_params, x, y)  # compute gradients
            updates, opt_state = optimizer.update(grads, opt_state)  # get updates
            params = optax.apply_updates(params, updates)
        return params, (0, {'a': 1, 'b': 2})

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        data = (xs, ys, fmodel)
        inner_params = pytree.tree_map(
            lambda t: t.clone().detach_().requires_grad_(requires_grad=t.requires_grad),
            params,
        )
        optimal_inner_params, aux = inner_solver_torchopt(inner_params, params, data)
        assert aux == (0, {'a': 1, 'b': 2})
        outer_loss = fmodel(optimal_inner_params, xs).mean()

        grads = torch.autograd.grad(outer_loss, params)
        updates, optim_state = optim.update(grads, optim_state)
        params = torchopt.apply_updates(params, updates)

        xs = xs.numpy()
        ys = ys.numpy()

        def outer_level(p, xs, ys):
            optimal_params, aux = inner_solver_jax(copy.deepcopy(p), p, xs, ys)
            assert aux == (0, {'a': 1, 'b': 2})
            return jax_model(optimal_params, xs).mean()

        grads_jax = jax.grad(outer_level, argnums=0)(jax_params, xs, ys)
        updates_jax, optim_state_jax = optim_jax.update(grads_jax, optim_state_jax)  # get updates
        jax_params = optax.apply_updates(jax_params, updates_jax)

    jax_params_as_tensor = tuple(
        nn.Parameter(torch.tensor(np.asarray(jax_params[j]), dtype=dtype)) for j in jax_params
    )

    helpers.assert_pytree_all_close(params, jax_params_as_tensor)


@pytest.mark.skipif(not HAS_JAX, reason='JAX is not installed')
@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-3, 1e-4],
    inner_lr=[2e-2, 2e-3],
    inner_update=[20, 50, 100],
    ns=[False, True],
)
def test_imaml_solve_inv(
    dtype: torch.dtype,
    lr: float,
    inner_lr: float,
    inner_update: int,
    ns: bool,
) -> None:
    np_dtype = helpers.dtype_torch2numpy(dtype)

    jax_model, jax_params = get_model_jax(dtype=np_dtype)
    model, loader = get_model_torch(device='cpu', dtype=dtype)

    fmodel, params = functorch.make_functional(model)
    optim = torchopt.sgd(lr)
    optim_state = optim.init(params)

    optim_jax = optax.sgd(lr)
    optim_state_jax = optim_jax.init(jax_params)

    def imaml_objective_torchopt(params, meta_params, data):
        x, y, f = data
        y_pred = f(params, x)
        regularization_loss = 0
        for p1, p2 in zip(params, meta_params):
            regularization_loss += 0.5 * torch.sum(torch.square(p1 - p2))
        return F.cross_entropy(y_pred, y) + regularization_loss

    @torchopt.diff.implicit.custom_root(
        functorch.grad(imaml_objective_torchopt, argnums=0),
        argnums=1,
        solve=torchopt.linear_solve.solve_inv(ns=ns),
    )
    def inner_solver_torchopt(params, meta_params, data):
        # Initial functional optimizer based on TorchOpt
        x, y, f = data
        optimizer = torchopt.sgd(lr=inner_lr)
        opt_state = optimizer.init(params)
        with torch.enable_grad():
            # Temporarily enable gradient computation for conducting the optimization
            for _ in range(inner_update):
                pred = f(params, x)
                loss = F.cross_entropy(pred, y)  # compute loss
                # Compute regularization loss
                regularization_loss = 0
                for p1, p2 in zip(params, meta_params):
                    regularization_loss += 0.5 * torch.sum(torch.square(p1 - p2))
                final_loss = loss + regularization_loss
                grads = torch.autograd.grad(final_loss, params)  # compute gradients
                updates, opt_state = optimizer.update(grads, opt_state, inplace=True)  # get updates
                params = torchopt.apply_updates(params, updates, inplace=True)
        return params

    def imaml_objective_jax(params, meta_params, x, y):
        y_pred = jax_model(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_pred, y))
        regularization_loss = 0
        for p1, p2 in zip(params.values(), meta_params.values()):
            regularization_loss += 0.5 * jnp.sum(jnp.square(p1 - p2))
        return loss + regularization_loss

    @jaxopt.implicit_diff.custom_root(
        jax.grad(imaml_objective_jax, argnums=0),
        solve=jaxopt.linear_solve.solve_normal_cg,
    )
    def inner_solver_jax(params, meta_params, x, y):
        """Solve ridge regression by conjugate gradient."""
        # Initial functional optimizer based on torchopt
        optimizer = optax.sgd(inner_lr)
        opt_state = optimizer.init(params)

        def compute_loss(params, meta_params, x, y):
            pred = jax_model(params, x)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, y))
            # Compute regularization loss
            regularization_loss = 0
            for p1, p2 in zip(params.values(), meta_params.values()):
                regularization_loss += 0.5 * jnp.sum(jnp.square(p1 - p2))
            return loss + regularization_loss

        for _ in range(inner_update):
            grads = jax.grad(compute_loss)(params, meta_params, x, y)  # compute gradients
            updates, opt_state = optimizer.update(grads, opt_state)  # get updates
            params = optax.apply_updates(params, updates)
        return params

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        data = (xs, ys, fmodel)
        inner_params = pytree.tree_map(
            lambda t: t.clone().detach_().requires_grad_(requires_grad=t.requires_grad),
            params,
        )
        optimal_inner_params = inner_solver_torchopt(inner_params, params, data)
        outer_loss = fmodel(optimal_inner_params, xs).mean()

        grads = torch.autograd.grad(outer_loss, params)
        updates, optim_state = optim.update(grads, optim_state)
        params = torchopt.apply_updates(params, updates)

        xs = xs.numpy()
        ys = ys.numpy()

        def outer_level(p, xs, ys):
            optimal_params = inner_solver_jax(copy.deepcopy(p), p, xs, ys)
            return jax_model(optimal_params, xs).mean()

        grads_jax = jax.grad(outer_level, argnums=0)(jax_params, xs, ys)
        updates_jax, optim_state_jax = optim_jax.update(grads_jax, optim_state_jax)  # get updates
        jax_params = optax.apply_updates(jax_params, updates_jax)

    jax_params_as_tensor = tuple(
        nn.Parameter(torch.tensor(np.asarray(jax_params[j]), dtype=dtype)) for j in jax_params
    )

    helpers.assert_pytree_all_close(params, jax_params_as_tensor)


@pytest.mark.skipif(not HAS_JAX, reason='JAX is not installed')
@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-3, 1e-4],
    inner_lr=[2e-2, 2e-3],
    inner_update=[20, 50, 100],
)
def test_imaml_module(dtype: torch.dtype, lr: float, inner_lr: float, inner_update: int) -> None:
    np_dtype = helpers.dtype_torch2numpy(dtype)

    jax_model, jax_params = get_model_jax(dtype=np_dtype)
    model, loader = get_model_torch(device='cpu', dtype=dtype)

    class InnerNet(ImplicitMetaGradientModule):
        def __init__(self, meta_model):
            super().__init__()
            self.meta_model = meta_model
            self.model = torchopt.module_clone(meta_model, by='deepcopy', detach_buffers=True)

        def forward(self, x):
            return self.model(x)

        def objective(self, x, y):
            y_pred = self.model(x)
            loss = F.cross_entropy(y_pred, y)
            regularization_loss = 0
            for p1, p2 in zip(self.parameters(), self.meta_parameters()):
                regularization_loss += 0.5 * torch.sum(torch.square(p1 - p2))
            return loss + regularization_loss

        def solve(self, x, y):
            params = tuple(self.parameters())
            optim_inner = torchopt.SGD(params, lr=inner_lr)
            with torch.enable_grad():
                # Temporarily enable gradient computation for conducting the optimization
                for _ in range(inner_update):
                    loss = self.objective(x, y)
                    optim_inner.zero_grad()
                    loss.backward(inputs=params)
                    optim_inner.step()

            return self, (0, {'a': 1, 'b': 2})

    outer_optim = torchopt.SGD(model.parameters(), lr)

    optim_jax = optax.sgd(lr)
    optim_state_jax = optim_jax.init(jax_params)

    def imaml_objective_jax(params, meta_params, x, y):
        y_pred = jax_model(params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_pred, y))
        regularization_loss = 0
        for p1, p2 in zip(params.values(), meta_params.values()):
            regularization_loss += 0.5 * jnp.sum(jnp.square(p1 - p2))
        return loss + regularization_loss

    @jaxopt.implicit_diff.custom_root(jax.grad(imaml_objective_jax, argnums=0), has_aux=True)
    def inner_solver_jax(params, meta_params, x, y):
        """Solve ridge regression by conjugate gradient."""
        # Initial functional optimizer based on torchopt
        optimizer = optax.sgd(inner_lr)
        opt_state = optimizer.init(params)

        def compute_loss(params, meta_params, x, y):
            pred = jax_model(params, x)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, y))
            # Compute regularization loss
            regularization_loss = 0
            for p1, p2 in zip(params.values(), meta_params.values()):
                regularization_loss += 0.5 * jnp.sum(jnp.square(p1 - p2))
            return loss + regularization_loss

        for _ in range(inner_update):
            grads = jax.grad(compute_loss)(params, meta_params, x, y)  # compute gradients
            updates, opt_state = optimizer.update(grads, opt_state)  # get updates
            params = optax.apply_updates(params, updates)
        return params, (0, {'a': 1, 'b': 2})

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        inner_model = InnerNet(model)
        optimal_inner_model, aux = inner_model.solve(xs, ys)
        assert aux == (0, {'a': 1, 'b': 2})
        outer_loss = optimal_inner_model(xs).mean()

        outer_optim.zero_grad()
        outer_loss.backward()
        outer_optim.step()

        xs = xs.numpy()
        ys = ys.numpy()

        def outer_level(p, xs, ys):
            optimal_params, aux = inner_solver_jax(copy.deepcopy(p), p, xs, ys)
            assert aux == (0, {'a': 1, 'b': 2})
            return jax_model(optimal_params, xs).mean()

        grads_jax = jax.grad(outer_level, argnums=0)(jax_params, xs, ys)
        updates_jax, optim_state_jax = optim_jax.update(grads_jax, optim_state_jax)  # get updates
        jax_params = optax.apply_updates(jax_params, updates_jax)

    jax_params_as_tensor = tuple(
        nn.Parameter(torch.tensor(np.asarray(jax_params[j]), dtype=dtype)) for j in jax_params
    )

    helpers.assert_pytree_all_close(tuple(model.parameters()), jax_params_as_tensor)


@pytest.mark.skipif(not HAS_JAX, reason='JAX is not installed')
@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-3, 1e-4],
)
def test_rr_solve_cg(
    dtype: torch.dtype,
    lr: float,
) -> None:
    helpers.seed_everything(42)
    np_dtype = helpers.dtype_torch2numpy(dtype)
    input_size = 10

    init_params_torch = torch.randn(input_size, dtype=dtype)
    l2reg_torch = torch.rand(1, dtype=dtype).squeeze_().requires_grad_(True)

    init_params_jax = jnp.array(init_params_torch.detach().numpy(), dtype=np_dtype)
    l2reg_jax = jnp.array(l2reg_torch.detach().numpy(), dtype=np_dtype)

    loader = get_rr_dataset_torch()

    optim = torchopt.sgd(lr)
    optim_state = optim.init(l2reg_torch)

    optim_jax = optax.sgd(lr)
    optim_state_jax = optim_jax.init(l2reg_jax)

    def ridge_objective_torch(params, l2reg, data):
        """Ridge objective function."""
        X_tr, y_tr = data
        residuals = X_tr @ params - y_tr
        regularization_loss = 0.5 * l2reg * torch.sum(torch.square(params))
        return 0.5 * torch.mean(torch.square(residuals)) + regularization_loss

    @torchopt.diff.implicit.custom_root(functorch.grad(ridge_objective_torch, argnums=0), argnums=1)
    def ridge_solver_torch_cg(params, l2reg, data):
        """Solve ridge regression by conjugate gradient."""
        X_tr, y_tr = data

        def matvec(u):
            return X_tr.T @ (X_tr @ u)

        solve = torchopt.linear_solve.solve_cg(
            ridge=len(y_tr) * l2reg.item(),
            init=params,
            maxiter=20,
        )

        return solve(matvec=matvec, b=X_tr.T @ y_tr)

    def ridge_objective_jax(params, l2reg, X_tr, y_tr):
        """Ridge objective function."""
        residuals = X_tr @ params - y_tr
        regularization_loss = 0.5 * l2reg * jnp.sum(jnp.square(params))
        return 0.5 * jnp.mean(jnp.square(residuals)) + regularization_loss

    @jaxopt.implicit_diff.custom_root(jax.grad(ridge_objective_jax, argnums=0))
    def ridge_solver_jax_cg(params, l2reg, X_tr, y_tr):
        """Solve ridge regression by conjugate gradient."""

        def matvec(u):
            return X_tr.T @ (X_tr @ u)

        return jaxopt.linear_solve.solve_cg(
            matvec=matvec,
            b=X_tr.T @ y_tr,
            ridge=len(y_tr) * l2reg.item(),
            init=params,
            maxiter=20,
        )

    for xs, ys, xq, yq in loader:
        xs = xs.to(dtype=dtype)
        ys = ys.to(dtype=dtype)
        xq = xq.to(dtype=dtype)
        yq = yq.to(dtype=dtype)

        w_fit = ridge_solver_torch_cg(init_params_torch, l2reg_torch, (xs, ys))
        outer_loss = F.mse_loss(xq @ w_fit, yq)

        grads, *_ = torch.autograd.grad(outer_loss, l2reg_torch)
        updates, optim_state = optim.update(grads, optim_state)
        l2reg_torch = torchopt.apply_updates(l2reg_torch, updates)

        xs = jnp.array(xs.numpy(), dtype=np_dtype)
        ys = jnp.array(ys.numpy(), dtype=np_dtype)
        xq = jnp.array(xq.numpy(), dtype=np_dtype)
        yq = jnp.array(yq.numpy(), dtype=np_dtype)

        def outer_level(params_jax, l2reg_jax, xs, ys, xq, yq):
            w_fit = ridge_solver_jax_cg(params_jax, l2reg_jax, xs, ys)
            y_pred = xq @ w_fit
            return jnp.mean(jnp.square(y_pred - yq))

        grads_jax = jax.grad(outer_level, argnums=1)(init_params_jax, l2reg_jax, xs, ys, xq, yq)
        updates_jax, optim_state_jax = optim_jax.update(grads_jax, optim_state_jax)  # get updates
        l2reg_jax = optax.apply_updates(l2reg_jax, updates_jax)

    l2reg_jax_as_tensor = torch.tensor(np.asarray(l2reg_jax), dtype=dtype)
    helpers.assert_all_close(l2reg_torch, l2reg_jax_as_tensor)


@pytest.mark.skipif(not HAS_JAX, reason='JAX is not installed')
@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-3, 1e-4],
    ns=[True, False],
)
def test_rr_solve_inv(
    dtype: torch.dtype,
    lr: float,
    ns: bool,
) -> None:
    if dtype == torch.float64 and ns:
        pytest.skip('Neumann Series test skips torch.float64 due to numerical stability.')
    helpers.seed_everything(42)
    np_dtype = helpers.dtype_torch2numpy(dtype)
    input_size = 10

    init_params_torch = torch.randn(input_size, dtype=dtype)
    l2reg_torch = torch.rand(1, dtype=dtype).squeeze_().requires_grad_(True)

    init_params_jax = jnp.array(init_params_torch.detach().numpy(), dtype=np_dtype)
    l2reg_jax = jnp.array(l2reg_torch.detach().numpy(), dtype=np_dtype)

    loader = get_rr_dataset_torch()

    optim = torchopt.sgd(lr)
    optim_state = optim.init(l2reg_torch)

    optim_jax = optax.sgd(lr)
    optim_state_jax = optim_jax.init(l2reg_jax)

    def ridge_objective_torch(params, l2reg, data):
        """Ridge objective function."""
        X_tr, y_tr = data
        residuals = X_tr @ params - y_tr
        regularization_loss = 0.5 * l2reg * torch.sum(torch.square(params))
        return 0.5 * torch.mean(torch.square(residuals)) + regularization_loss

    @torchopt.diff.implicit.custom_root(functorch.grad(ridge_objective_torch, argnums=0), argnums=1)
    def ridge_solver_torch_inv(params, l2reg, data):
        """Solve ridge regression by conjugate gradient."""
        X_tr, y_tr = data

        def matvec(u):
            return X_tr.T @ (X_tr @ u)

        solve = torchopt.linear_solve.solve_inv(
            matvec=matvec,
            b=X_tr.T @ y_tr,
            ridge=len(y_tr) * l2reg.item(),
            ns=ns,
        )

        return solve(matvec=matvec, b=X_tr.T @ y_tr)

    def ridge_objective_jax(params, l2reg, X_tr, y_tr):
        """Ridge objective function."""
        residuals = X_tr @ params - y_tr
        regularization_loss = 0.5 * l2reg * jnp.sum(jnp.square(params))
        return 0.5 * jnp.mean(jnp.square(residuals)) + regularization_loss

    @jaxopt.implicit_diff.custom_root(jax.grad(ridge_objective_jax, argnums=0))
    def ridge_solver_jax_inv(params, l2reg, X_tr, y_tr):
        """Solve ridge regression by conjugate gradient."""

        def matvec(u):
            return X_tr.T @ (X_tr @ u)

        return jaxopt.linear_solve.solve_inv(
            matvec=matvec,
            b=X_tr.T @ y_tr,
            ridge=len(y_tr) * l2reg.item(),
        )

    for xs, ys, xq, yq in loader:
        xs = xs.to(dtype=dtype)
        ys = ys.to(dtype=dtype)
        xq = xq.to(dtype=dtype)
        yq = yq.to(dtype=dtype)

        w_fit = ridge_solver_torch_inv(init_params_torch, l2reg_torch, (xs, ys))
        outer_loss = F.mse_loss(xq @ w_fit, yq)

        grads, *_ = torch.autograd.grad(outer_loss, l2reg_torch)
        updates, optim_state = optim.update(grads, optim_state)
        l2reg_torch = torchopt.apply_updates(l2reg_torch, updates)

        xs = jnp.array(xs.numpy(), dtype=np_dtype)
        ys = jnp.array(ys.numpy(), dtype=np_dtype)
        xq = jnp.array(xq.numpy(), dtype=np_dtype)
        yq = jnp.array(yq.numpy(), dtype=np_dtype)

        def outer_level(params_jax, l2reg_jax, xs, ys, xq, yq):
            w_fit = ridge_solver_jax_inv(params_jax, l2reg_jax, xs, ys)
            y_pred = xq @ w_fit
            return jnp.mean(jnp.square(y_pred - yq))

        grads_jax = jax.grad(outer_level, argnums=1)(init_params_jax, l2reg_jax, xs, ys, xq, yq)
        updates_jax, optim_state_jax = optim_jax.update(grads_jax, optim_state_jax)  # get updates
        l2reg_jax = optax.apply_updates(l2reg_jax, updates_jax)

    l2reg_jax_as_tensor = torch.tensor(np.asarray(l2reg_jax), dtype=dtype)
    helpers.assert_all_close(l2reg_torch, l2reg_jax_as_tensor)


def test_module_empty_parameters() -> None:
    class EmptyParameters(ImplicitMetaGradientModule):
        def __init__(self, x):
            super().__init__()
            self.x = x

        def objective(self):
            return self.x.mean()

        def solve(self):
            pass

    model = EmptyParameters(torch.zeros(8))
    with pytest.raises(RuntimeError, match='The module has no parameters.'):
        model.solve()

    model = EmptyParameters(torch.zeros(8))
    model.register_parameter('y', torch.zeros(8, requires_grad=True))
    with pytest.raises(RuntimeError, match='The module has no meta-parameters.'):
        model.solve()

    model = EmptyParameters(torch.zeros(8, requires_grad=True))
    with pytest.raises(RuntimeError, match='The module has no parameters.'):
        model.solve()

    model = EmptyParameters(torch.zeros(8, requires_grad=True))
    with pytest.raises(RuntimeError, match='The module has no parameters.'):
        model.optimality()

    model = EmptyParameters(torch.zeros(8))
    model.register_parameter('y', torch.zeros(8, requires_grad=True))
    with pytest.raises(RuntimeError, match='The module has no meta-parameters.'):
        model.optimality()

    model = EmptyParameters(torch.zeros(8, requires_grad=True))
    model.register_parameter('y', torch.zeros(8, requires_grad=True))
    model.solve()

    model = EmptyParameters(nn.Linear(8, 8).eval())
    with pytest.raises(RuntimeError, match='The module has no meta-parameters.'):
        model.solve()

    model = EmptyParameters(nn.Linear(8, 8))
    model.register_parameter('y', torch.zeros(8, requires_grad=True))
    model.solve()


def test_module_enable_implicit_gradients_twice() -> None:
    class MyModule1(torchopt.nn.ImplicitMetaGradientModule):
        def objective(self):
            return torch.tensor(0.0)

        def solve(self):
            pass

    from torchopt.diff.implicit.nn.module import (
        enable_implicit_gradients,
        make_optimality_from_objective,
    )

    with pytest.raises(
        TypeError,
        match='Implicit gradients are already enabled for the `solve` method.',
    ):
        enable_implicit_gradients(MyModule1)

    class MyModule2(torchopt.nn.ImplicitMetaGradientModule):
        def optimality(self):
            return torch.tensor(0.0)

        def solve(self):
            pass

    with pytest.raises(
        TypeError,
        match='The objective function is not defined.',
    ):
        make_optimality_from_objective(MyModule2)


def test_module_abstract_methods() -> None:
    class MyModule1(torchopt.nn.ImplicitMetaGradientModule):
        def objective(self):
            return torch.tensor(0.0)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        MyModule1()

    with pytest.raises(
        TypeError,
        match=re.escape(
            'ImplicitMetaGradientModule requires either an optimality() method or an objective() method',
        ),
    ):

        class MyModule2(torchopt.nn.ImplicitMetaGradientModule):
            def solve(self):
                pass

    class MyModule3(torchopt.nn.ImplicitMetaGradientModule):
        def optimality(self):
            return ()

        def solve(self):
            pass

    with pytest.raises(
        TypeError,
        match=re.escape('method optimality() must not be a staticmethod.'),
    ):

        class MyModule4(torchopt.nn.ImplicitMetaGradientModule):
            @staticmethod
            def optimality():
                return ()

            def solve(self):
                pass

    with pytest.raises(
        TypeError,
        match=re.escape('method optimality() must not be a classmethod.'),
    ):

        class MyModule5(torchopt.nn.ImplicitMetaGradientModule):
            @classmethod
            def optimality(self):
                return ()

            def solve(self):
                pass

    with pytest.raises(
        TypeError,
        match=re.escape('method optimality() must be callable.'),
    ):

        class MyModule6(torchopt.nn.ImplicitMetaGradientModule):
            optimality = 0

            def solve(self):
                pass

    with pytest.raises(
        TypeError,
        match=re.escape('method objective() must not be a staticmethod.'),
    ):

        class MyModule7(torchopt.nn.ImplicitMetaGradientModule):
            @staticmethod
            def objective():
                return ()

            def solve(self):
                pass

    with pytest.raises(
        TypeError,
        match=re.escape('method objective() must not be a classmethod.'),
    ):

        class MyModule8(torchopt.nn.ImplicitMetaGradientModule):
            @classmethod
            def objective(self):
                return ()

            def solve(self):
                pass

    with pytest.raises(
        TypeError,
        match=re.escape('method objective() must be callable.'),
    ):

        class MyModule9(torchopt.nn.ImplicitMetaGradientModule):
            objective = 0

            def solve(self):
                pass
