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
from typing import Optional, Tuple, Union

import functorch
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import optax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import helpers
import torchopt
from torchopt import pytree


BATCH_SIZE = 8
NUM_UPDATES = 3

MODEL_NUM_INPUTS = 10
MODEL_NUM_CLASSES = 10


def get_model_jax(dtype: np.dtype = np.float32) -> Tuple[FunctionType, OrderedDict]:
    helpers.seed_everything(seed=42)

    def func(params, x):
        return jnp.matmul(x, params['weight']) + params['bias']

    params = OrderedDict(
        [
            ('weight', jnp.ones((MODEL_NUM_INPUTS, MODEL_NUM_CLASSES), dtype=dtype)),
            ('bias', jnp.zeros((MODEL_NUM_CLASSES,), dtype=dtype)),
        ]
    )
    return func, params


@torch.no_grad()
def get_model_torch(
    device: Optional[Union[str, torch.device]] = None, dtype: torch.dtype = torch.float32
) -> Tuple[nn.Module, data.DataLoader]:
    helpers.seed_everything(seed=42)

    class FcNet(nn.Module):
        def __init__(self, dim, out):
            super().__init__()
            self.fc = nn.Linear(in_features=dim, out_features=out, bias=True)
            nn.init.ones_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        def forward(self, x):
            return self.fc(x)

    model = FcNet(MODEL_NUM_INPUTS, MODEL_NUM_CLASSES).to(dtype=dtype)

    if device is not None:
        model = model.to(device=torch.device(device))

    dataset = data.TensorDataset(
        torch.randint(0, 1, (BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS)),
        torch.randint(0, MODEL_NUM_CLASSES, (BATCH_SIZE * NUM_UPDATES,)),
    )
    loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=False)

    return model, loader


@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-3, 1e-4],
    inner_lr=[2e-2, 2e-3],
    inner_update=[20, 50, 100],
)
def test_imaml(dtype: torch.dtype, lr: float, inner_lr: float, inner_update: int) -> None:
    jax_function, jax_params = get_model_jax(dtype=helpers.dtype_torch2numpy(dtype))
    model, loader = get_model_torch(device='cpu', dtype=dtype)

    fmodel, params = functorch.make_functional(model)
    optim = torchopt.sgd(lr)
    optim_state = optim.init(params)

    optim_jax = optax.sgd(lr)
    optim_state_jax = optim_jax.init(jax_params)

    def imaml_objective_torchopt(optimal_params, init_params, data):
        x, y, f = data
        y_pred = f(optimal_params, x)
        regularization_loss = 0
        for p1, p2 in zip(optimal_params, init_params):
            regularization_loss += 0.5 * torch.sum(torch.square(p1 - p2))
        loss = F.cross_entropy(y_pred, y) + regularization_loss
        return loss

    @torchopt.implicit_diff.custom_root(
        functorch.grad(imaml_objective_torchopt, argnums=0), argnums=1
    )
    def inner_solver_torchopt(init_params_copy, init_params, data):
        # Initial functional optimizer based on TorchOpt
        x, y, f = data
        params = init_params_copy
        optimizer = torchopt.sgd(lr=inner_lr)
        opt_state = optimizer.init(params)
        with torch.enable_grad():
            # Temporarily enable gradient computation for conducting the optimization
            for _ in range(inner_update):
                pred = f(params, x)
                loss = F.cross_entropy(pred, y)  # compute loss
                # Compute regularization loss
                regularization_loss = 0
                for p1, p2 in zip(params, init_params):
                    regularization_loss += 0.5 * torch.sum(torch.square(p1 - p2))
                final_loss = loss + regularization_loss
                grads = torch.autograd.grad(final_loss, params)  # compute gradients
                updates, opt_state = optimizer.update(grads, opt_state)  # get updates
                params = torchopt.apply_updates(params, updates)
        return params

    def imaml_objective_jax(optimal_params, init_params, x, y):
        y_pred = jax_function(optimal_params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_pred, y))
        regularization_loss = 0
        for p1, p2 in zip(optimal_params.values(), init_params.values()):
            regularization_loss += 0.5 * jnp.sum(jnp.square((p1 - p2)))
        loss = loss + regularization_loss
        return loss

    @jaxopt.implicit_diff.custom_root(jax.grad(imaml_objective_jax, argnums=0))
    def inner_solver_jax(init_params_copy, init_params, x, y):
        """Solve ridge regression by conjugate gradient."""
        # Initial functional optimizer based on torchopt
        params = init_params_copy
        optimizer = optax.sgd(inner_lr)
        opt_state = optimizer.init(params)

        def compute_loss(params, init_params, x, y):
            pred = jax_function(params, x)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, y))
            # Compute regularization loss
            regularization_loss = 0
            for p1, p2 in zip(params.values(), init_params.values()):
                regularization_loss += 0.5 * jnp.sum(jnp.square((p1 - p2)))
            final_loss = loss + regularization_loss
            return final_loss

        for i in range(inner_update):
            grads = jax.grad(compute_loss)(params, init_params, x, y)  # compute gradients
            updates, opt_state = optimizer.update(grads, opt_state)  # get updates
            params = optax.apply_updates(params, updates)
        return params

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        data = (xs, ys, fmodel)
        init_params_copy = pytree.tree_map(
            lambda t: t.clone().detach_().requires_grad_(requires_grad=t.requires_grad), params
        )
        optimal_params = inner_solver_torchopt(init_params_copy, params, data)
        outer_loss = fmodel(optimal_params, xs).mean()

        grad = torch.autograd.grad(outer_loss, params)
        updates, optim_state = optim.update(grad, optim_state)
        params = torchopt.apply_updates(params, updates)

        xs = xs.numpy()
        ys = ys.numpy()

        def outer_level(p, xs, ys):
            optimal_params = inner_solver_jax(copy.deepcopy(p), p, xs, ys)
            outer_loss = jax_function(optimal_params, xs).mean()
            return outer_loss

        grads_jax = jax.grad(outer_level, argnums=0)(jax_params, xs, ys)
        updates_jax, optim_state_jax = optim_jax.update(grads_jax, optim_state_jax)  # get updates
        jax_params = optax.apply_updates(jax_params, updates_jax)

    jax_params_as_tensor = tuple(
        nn.Parameter(torch.tensor(np.asarray(jax_params[j]), dtype=dtype)) for j in jax_params
    )

    for p, p_ref in zip(params, jax_params_as_tensor):
        helpers.assert_all_close(p, p_ref)
        

@torch.no_grad()
def get_dataset_torch_rr(
    device: Optional[Union[str, torch.device]] = None) -> Tuple[nn.Module, data.DataLoader]:
    helpers.seed_everything(seed=42)
    NUM_UPDATES = 4
    BATCH_SIZE = 1024
    dataset = data.TensorDataset(
        torch.randn((BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS)),
        torch.randn((BATCH_SIZE * NUM_UPDATES)),
        torch.randn((BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS)),
        torch.randn((BATCH_SIZE * NUM_UPDATES))
    )
    loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=False)

    return loader

@helpers.parametrize(
    lr=[1e-3, 1e-4],
    dtype=[torch.float64],
    jax_dtype=[jnp.float64],
)
def test_rr(
    lr: float,
    dtype: torch.dtype,
    jax_dtype: jnp.dtype,
) -> None:
    helpers.seed_everything(42)
    device = 'cpu'
    input_size = 10
    
    init_params_torch = torch.randn(input_size).to(device, dtype=dtype)
    l2reg_torch = torch.rand(1, requires_grad=True).to(device, dtype=dtype)
    
    init_params_jax = jnp.array(init_params_torch.detach().numpy(), dtype=jax_dtype)
    l2reg_jax = jnp.array(l2reg_torch.detach().numpy(), dtype=jax_dtype)
    
    loader = get_dataset_torch_rr(device='cpu')

    optim = torchopt.sgd(lr)
    optim_state = optim.init(l2reg_torch)
    
    optim_jax = optax.sgd(lr)
    opt_state_jax = optim_jax.init(l2reg_jax)
    
    def ridge_objective_torch(params, l2reg, data):
        """Ridge objective function."""
        x_tr, y_tr = data
        params = params
        residuals  = x_tr @ params - y_tr
        return 0.5 * torch.mean(residuals ** 2) + 0.5 * l2reg.sum() * torch.sum(params ** 2)

    @torchopt.implicit_diff.custom_root(functorch.grad(ridge_objective_torch, argnums=0), argnums=1)
    def ridge_solver_torch(init_params, l2reg, data):
        """Solve ridge regression by conjugate gradient."""
        X_tr, y_tr = data

        def matvec(u):
            return torch.matmul(X_tr.T, torch.matmul(X_tr, u))

        return torchopt.linear_solve.solve_cg(matvec=matvec,
                                      b=torch.matmul(X_tr.T, y_tr),
                                      ridge=len(y_tr) * l2reg.item(),
                                      init=init_params,
                                      maxiter=20)
    
    def ridge_objective_jax(params, l2reg, X_tr, y_tr):
        """Ridge objective function."""
        #X_tr, y_tr = data
        residuals = jnp.dot(X_tr, params) - y_tr
        return 0.5 * jnp.mean(residuals ** 2) + 0.5 * jnp.sum(l2reg) * jnp.sum(params ** 2)


    @jaxopt.implicit_diff.custom_root(jax.grad(ridge_objective_jax, argnums=0))
    def ridge_solver_jax(init_params, l2reg, X_tr, y_tr):
      """Solve ridge regression by conjugate gradient."""

      def matvec(u):
        return jnp.dot(X_tr.T, jnp.dot(X_tr, u))

      return jaxopt.linear_solve.solve_cg(matvec=matvec,
                                   b=jnp.dot(X_tr.T, y_tr),
                                   ridge=len(y_tr) * l2reg.item(),
                                   init=init_params,
                                   maxiter=20)
    
    for xs, ys, xq, yq in loader:
        xs = xs.to(dtype=dtype)
        ys = ys.to(dtype=dtype)
        xq = xq.to(dtype=dtype)
        yq = yq.to(dtype=dtype)
        
        data = (xs, ys)
        #print(init_params_torch.shape, l2reg_torch.shape, xs.shape, ys.shape)
        w_fit = ridge_solver_torch(init_params_torch, l2reg_torch, (xs, ys))
        outer_loss = F.mse_loss(xq @ w_fit, yq)
        
        grad = torch.autograd.grad(outer_loss, l2reg_torch)[0]
        updates, optim_state = optim.update(grad, optim_state)
        l2reg_torch = torchopt.apply_updates(l2reg_torch, updates)
        
        xs = jnp.array(xs.numpy(), dtype=jax_dtype)
        ys = jnp.array(ys.numpy(), dtype=jax_dtype)
        xq = jnp.array(xq.numpy(), dtype=jax_dtype)
        yq = jnp.array(yq.numpy(), dtype=jax_dtype)
        
        def outer_level(init_params_jax, l2reg_jax, xs, ys, xq, yq):
            w_fit = ridge_solver_jax(init_params_jax, l2reg_jax, xs, ys)
            y_pred = jnp.dot(xq, w_fit)
            loss_value = jnp.mean((y_pred - yq) ** 2)
            return loss_value
        
        grads_jax = jax.grad(outer_level, argnums=1)(init_params_jax, l2reg_jax, xs, ys, xq, yq)
        updates_jax, opt_state_jax = optim_jax.update(grads_jax, opt_state_jax)  # get updates
        jax_params = optax.apply_updates(l2reg_jax, updates_jax)
    
    jax_p = torch.tensor(np.array(jax_params)).to(dtype=dtype)
    helpers.assert_all_close(l2reg_torch, jax_p, rtol=5e-5, atol=5e-5)
