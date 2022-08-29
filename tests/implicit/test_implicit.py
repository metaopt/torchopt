from typing import Tuple

import functorch
import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchopt

import jax
import jax.numpy as jnp
import jaxopt
import optax
import numpy as np

import helper_il as helpers
import copy

@helpers.parametrize(
    dtype=[torch.float32],
    lr=[1e-3, 1e-4],
    inner_update=[20, 50, 100]
)
def test_imaml(
    dtype: torch.dtype,
    lr: float,
    inner_update: int
) -> None:
    #if nesterov and momentum <= 0.0:
    #    pytest.skip('Nesterov momentum requires a momentum and zero dampening.')

    jax_function, jax_params = helpers.get_model_jax()
    model, loader = helpers.get_model_torch(device='cpu', dtype=dtype)

    fmodel, params = functorch.make_functional(model)
    optim = torchopt.sgd(lr)
    optim_state = optim.init(params)
    
    optim_jax = optax.sgd(lr)
    opt_state_jax = optim_jax.init(jax_params)
    
    def imaml_objective_torch(optimal_params, init_params, data):
        x, y, f = data
        y_pred = f(optimal_params, x)
        regularisation_loss = 0
        for p1, p2 in zip(optimal_params, init_params):
            regularisation_loss += 0.5 * torch.sum((p1.view(-1) - p2.view(-1))**2)
        loss = F.cross_entropy(y_pred, y) + regularisation_loss
        return loss 
    
    @torchopt.implicit_diff.custom_root(functorch.grad(imaml_objective_torch, argnums=0), argnums=1)
    def inner_solver_torch(init_params_copy, init_params, data):
        # inital functional optimizer based on torchopt
        x, y, f = data
        params = init_params_copy
        optimizer = torchopt.sgd(lr=2e-2)
        opt_state = optimizer.init(params)
        with torch.enable_grad():
            # temporarily enable gradient computation for conducting the optimization
            for i in range(inner_update):
                pred = f(params, x)   
                loss = F.cross_entropy(pred, y)                         # compute loss
                regularisation_loss = 0
                # compute regularisation loss
                for p1, p2 in zip(params, init_params):
                    regularisation_loss += 0.5 * torch.sum((p1.view(-1) - p2.view(-1))**2)
                final_loss = loss + regularisation_loss
                grads = torch.autograd.grad(final_loss, params)                # compute gradients
                updates, opt_state = optimizer.update(grads, opt_state)  # get updates
                params = torchopt.apply_updates(params, updates)       
        return params
    
    def imaml_objective(optimal_params, init_params, x, y):
        y_pred = jax_function(optimal_params, x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(y_pred, y))
        regularisation_loss = 0
        for p1, p2 in zip(optimal_params, init_params):
            regularisation_loss += 0.5 * jnp.sum((optimal_params[p1].reshape(-1) - init_params[p2].reshape(-1))**2)
        loss = loss + regularisation_loss
        return loss
    
    @jaxopt.implicit_diff.custom_root(jax.grad(imaml_objective, argnums=0))
    def inner_solver_jax(init_params_copy, init_params, x, y):
        """Solve ridge regression by conjugate gradient."""
        # inital functional optimizer based on torchopt
        #x, y, f = data
        params = init_params_copy
        optimizer = optax.sgd(2e-2)
        opt_state = optimizer.init(params)

        def compute_loss(params, init_params, x, y):
            pred = jax_function(params, x)
            #print(pred.shape)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(pred, y))
            regularisation_loss = 0
            # compute regularisation loss
            for p1, p2 in zip(params, init_params):
                regularisation_loss += 0.5 * jnp.sum((params[p1].reshape(-1) - init_params[p2].reshape(-1))**2)
            final_loss = loss + regularisation_loss
            #print(final_loss.shape)
            return final_loss

        for i in range(inner_update):                    
            grads = jax.grad(compute_loss)(params, init_params, x, y)                # compute gradients
            updates, opt_state = optimizer.update(grads, opt_state)  # get updates
            params = optax.apply_updates(params, updates)
        return params
    
    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        data = (xs, ys, fmodel)
        optimal_params = inner_solver_torch(helpers.clone(params), params, data)
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
        updates_jax, opt_state_jax = optim_jax.update(grads_jax, opt_state_jax)  # get updates
        jax_params = optax.apply_updates(jax_params, updates_jax)
        
    jax_p = tuple([nn.Parameter(torch.tensor(np.array(jax_params[j]))) for j in jax_params])
        
    helpers.assert_all_close(params, jax_p)