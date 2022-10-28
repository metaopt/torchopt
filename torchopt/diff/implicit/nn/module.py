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
"""The base class for differentiable implicit meta-gradient models."""

import functools
import itertools
from typing import Callable, Dict, Optional, Tuple

import functorch
import torch

import torchopt.nn
from torchopt import pytree
from torchopt.diff.implicit.decorator import custom_root
from torchopt.typing import TensorTree, TupleOfTensors  # pylint: disable=unused-import
from torchopt.utils import extract_module_containers


__all__ = ['ImplicitMetaGradientModule']


def make_residual_from_objective(
    objective: Callable[..., torch.Tensor]
) -> Callable[..., TupleOfTensors]:
    """Make a function that computes the optimality residual of the objective function."""
    # pylint: disable-next=redefined-builtin
    def residual(self: 'ImplicitMetaGradientModule', *input, **kwargs) -> TupleOfTensors:
        params_containers = extract_module_containers(self, with_buffers=False)[0]
        params_containers_backups = [container.copy() for container in params_containers]
        flat_params: TupleOfTensors
        # pylint: disable-next=line-too-long
        flat_params, params_containers_treespec = pytree.tree_flatten_as_tuple(params_containers)  # type: ignore[arg-type]

        # pylint: disable-next=redefined-builtin
        def objective_fn(flat_params: TupleOfTensors, *input, **kwargs) -> torch.Tensor:
            flat_grad_tracking_params = flat_params
            grad_tracking_params_containers: Tuple[
                Dict[str, Optional[torch.Tensor]], ...
            ] = pytree.tree_unflatten(  # type: ignore[assignment]
                params_containers_treespec, flat_grad_tracking_params
            )

            try:
                for container, grad_tracking_container in zip(
                    params_containers, grad_tracking_params_containers
                ):
                    container.update(grad_tracking_container)

                return objective(self, *input, **kwargs)
            finally:
                for container, container_backup in zip(
                    params_containers, params_containers_backups
                ):
                    container.update(container_backup)

        objective_grad_fn = functorch.grad(objective_fn, argnums=0)
        flat_grads = objective_grad_fn(flat_params, *input, **kwargs)
        return flat_grads

    return residual


def enable_implicit_gradients(
    solve: Callable[..., 'ImplicitMetaGradientModule']
) -> Callable[..., 'ImplicitMetaGradientModule']:
    """Enable implicit gradients for the :func:`solve` function."""
    if getattr(solve, '__implicit_gradients_enabled__', False):
        raise ValueError('Implicit gradients are already enabled for the solve function.')

    @functools.wraps(solve)
    def wrapped(  # pylint: disable=too-many-locals
        self: 'ImplicitMetaGradientModule', *input, **kwargs  # pylint: disable=redefined-builtin
    ) -> 'ImplicitMetaGradientModule':
        """Solve the optimization problem."""
        params_containers = extract_module_containers(self, with_buffers=False)[0]
        meta_params_containers = [self._meta_parameters]  # pylint: disable=protected-access
        for meta_module in self.meta_children():
            meta_params_containers.extend(
                extract_module_containers(meta_module, with_buffers=False)[0]
            )
        meta_params_containers = tuple(meta_params_containers)
        params_containers_backups = tuple(container.copy() for container in params_containers)
        meta_params_containers_backups = tuple(
            container.copy() for container in meta_params_containers
        )

        flat_params: TupleOfTensors
        flat_meta_params: TupleOfTensors
        flat_params, params_containers_treespec = pytree.tree_flatten_as_tuple(
            params_containers  # type: ignore[arg-type]
        )
        flat_meta_params, meta_params_containers_treespec = pytree.tree_flatten_as_tuple(
            meta_params_containers  # type: ignore[arg-type]
        )

        def optimality_fn(
            flat_params: TupleOfTensors,
            flat_meta_params: TupleOfTensors,
            *input,  # pylint: disable=redefined-builtin
            **kwargs,
        ) -> TupleOfTensors:
            flat_grad_tracking_params = flat_params
            grad_tracking_params_containers: Tuple[
                Dict[str, Optional[torch.Tensor]], ...
            ] = pytree.tree_unflatten(  # type: ignore[assignment]
                params_containers_treespec, flat_grad_tracking_params
            )
            flat_grad_tracking_meta_params = flat_meta_params
            grad_tracking_meta_params_containers: Tuple[
                Dict[str, Optional[torch.Tensor]], ...
            ] = pytree.tree_unflatten(  # type: ignore[assignment]
                meta_params_containers_treespec, flat_grad_tracking_meta_params
            )

            try:
                for container, grad_tracking_container in itertools.chain(
                    zip(params_containers, grad_tracking_params_containers),
                    zip(meta_params_containers, grad_tracking_meta_params_containers),
                ):
                    container.update(grad_tracking_container)

                return self.residual(*input, **kwargs)
            finally:
                for container, container_backup in itertools.chain(
                    zip(params_containers, params_containers_backups),
                    zip(meta_params_containers, meta_params_containers_backups),
                ):
                    container.update(container_backup)

        @custom_root(optimality_fn, argnums=1)
        def solve_fn(
            flat_params: TupleOfTensors,  # pylint: disable=unused-argument
            flat_meta_params: TupleOfTensors,  # pylint: disable=unused-argument
            *input,  # pylint: disable=redefined-builtin
            **kwargs,
        ) -> TupleOfTensors:
            solve(self, *input, **kwargs)
            return tuple(pytree.tree_leaves(params_containers))  # type: ignore[arg-type]

        # pylint: disable-next=unused-variable
        flat_optimal_params = solve_fn(flat_params, flat_meta_params, *input, **kwargs)
        return self

    wrapped.__implicit_gradients_enabled__ = True  # type: ignore[attr-defined]
    return wrapped


class ImplicitMetaGradientModule(torchopt.nn.MetaGradientModule):
    """The base class for differentiable implicit meta-gradient models."""

    _custom_residual: bool
    _custom_objective: bool

    def __init_subclass__(cls) -> None:
        """Initialize the subclass."""
        super().__init_subclass__()

        residual = getattr(cls, 'residual', ImplicitMetaGradientModule.residual)
        objective = getattr(cls, 'objective', ImplicitMetaGradientModule.objective)
        cls._custom_residual = residual is not ImplicitMetaGradientModule.residual
        cls._custom_objective = objective is not ImplicitMetaGradientModule.objective

        if cls._custom_residual:
            if isinstance(residual, staticmethod):
                raise TypeError('residual() must not be a staticmethod.')
            if isinstance(residual, classmethod):
                raise TypeError('residual() must not be a classmethod.')
            if not callable(residual):
                raise TypeError('residual() must be callable.')
        elif not cls._custom_objective:
            raise TypeError(
                'ImplicitMetaGradientModule requires either an residual() or an objective() function'
            )
        else:
            if isinstance(objective, staticmethod):
                raise TypeError('objective() must not be a staticmethod.')
            if isinstance(objective, classmethod):
                raise TypeError('objective() must not be a classmethod.')
            if not callable(objective):
                raise TypeError('objective() must be callable.')

            cls.residual = make_residual_from_objective(objective)  # type: ignore[assignment]

        cls.solve = enable_implicit_gradients(cls.solve)  # type: ignore[assignment]

    # pylint: disable-next=redefined-builtin
    def solve(self, *input, **kwargs) -> 'ImplicitMetaGradientModule':
        """Solves the inner optimization problem.

        .. warning::

            For gradient-based optimization methods, the parameter inputs should be explicitly
            specified in the :func:`torch.autograd.backward` function as argument ``inputs``.
            Otherwise, if not provided, the gradient is accumulated into all the leaf Tensors
            (including the meta-parameters) that were used to compute the objective output.
            Alternatively, please use :func:`torch.autograd.grad` instead.

        Example::

            def solve(self, batch, labels):
                parameters = tuple(self.parameters())
                optimizer = torch.optim.Adam(parameters, lr=1e-3)
                with torch.enable_grad():
                    for _ in range(100):
                        loss = self.objective(batch, labels)
                        optimizer.zero_grad()
                        # Only update the `.grad` attribute for parameters
                        # and leave the meta-parameters unchanged
                        loss.backward(inputs=parameters)
                        optimizer.step()
                return self

        Returns:
            The module itself after solving the inner optimization problem.
        """
        raise NotImplementedError  # update parameters

    # pylint: disable-next=redefined-builtin
    def residual(self, *input, **kwargs) -> 'TensorTree':
        r"""Computes the optimality residual.

        This method stands for the residual to the optimal parameters after solving the inner
        optimization problem (:meth:`solve`), i.e.:

        .. code-block:: python

            module.solve(*input, **kwargs)
            module.residual(*input, **kwargs)  # -> 0

        1. For gradient-based optimization, the :meth:`residual` function is the KKT condition,
        usually it is the gradients of the :meth:`objective` function with respect to the module
        parameters (not the meta-parameters). If this method is not implemented, it will be
        automatically derived from the gradient of the :meth:`objective` function.

        .. math::

            \text{residual} = \nabla_{\boldsymbol{x}} f (\boldsymbol{x}, \boldsymbol{\theta}) \to \boldsymbol{0}

        where :math:`\boldsymbol{x}` is the joint vector of the module parameters and
        :math:`\boldsymbol{\theta}` is the joint vector of the meta-parameters.

        References:
            - Karush-Kuhn-Tucker (KKT) conditions: https://en.wikipedia.org/wiki/Karush-Kuhn-Tucker_conditions

        2. For fixed point iteration, the :meth:`residual` function can be the residual of the
        parameters between iterations, i.e.:

        .. math::

            \text{residual} = f (\boldsymbol{x}, \boldsymbol{\theta}) - \boldsymbol{x} \to \boldsymbol{0}

        where :math:`\boldsymbol{x}` is the joint vector of the module parameters and
        :math:`\boldsymbol{\theta}` is the joint vector of the meta-parameters.

        Returns:
            A tree of tensors, the residual to the optimal parameters after solving the inner
            optimization problem.
        """
        raise NotImplementedError

    # pylint: disable-next=redefined-builtin
    def objective(self, *input, **kwargs) -> torch.Tensor:
        """Computes the objective function value.

        This method is used to calculate the :meth:`residual` if it is not implemented.
        Otherwise, this method is optional.

        Returns:
            A scalar tensor (``dim=0``), the objective function value.
        """
        raise NotImplementedError
