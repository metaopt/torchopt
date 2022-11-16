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

# pylint: disable=redefined-builtin

import contextlib
import functools
import itertools
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple, Type

import functorch
import torch

from torchopt import pytree
from torchopt.diff.implicit.decorator import custom_root
from torchopt.nn.module import MetaGradientModule
from torchopt.typing import LinearSolver, TensorTree, TupleOfTensors
from torchopt.utils import extract_module_containers


__all__ = ['ImplicitMetaGradientModule']


def update_containers(
    dst_containers: Iterable[Dict[str, Optional[torch.Tensor]]],
    src_containers: Iterable[Dict[str, Optional[torch.Tensor]]],
) -> None:
    """Update the tensor containers in ``dst_containers`` with the ones in ``src_containers``."""
    for src_container, dst_container in zip(src_containers, dst_containers):
        dst_container.update(src_container)


@contextlib.contextmanager
def container_context(
    orig_containers: Iterable[Dict[str, Optional[torch.Tensor]]],
    args_containers: Iterable[Dict[str, Optional[torch.Tensor]]],
) -> Generator[None, None, None]:
    # pylint: disable-next=line-too-long
    """A context manager that temporarily updates the containers in ``orig_containers`` with the ones in ``args_containers``."""
    if not isinstance(orig_containers, (list, tuple)):
        orig_containers = list(orig_containers)
    orig_containers_backups = [container.copy() for container in orig_containers]
    try:
        update_containers(orig_containers, args_containers)
        yield
    finally:
        update_containers(orig_containers, orig_containers_backups)


def make_optimality_from_objective(
    objective: Callable[..., torch.Tensor]
) -> Callable[..., TupleOfTensors]:
    """Make a function that computes the optimality function of the objective function."""

    def optimality(self: 'ImplicitMetaGradientModule', *input, **kwargs) -> TupleOfTensors:
        params_containers = extract_module_containers(self, with_buffers=False)[0]
        flat_params: TupleOfTensors
        # pylint: disable-next=line-too-long
        flat_params, params_containers_treespec = pytree.tree_flatten_as_tuple(params_containers)  # type: ignore[arg-type]

        def objective_fn(__flat_params: TupleOfTensors, *input, **kwargs) -> torch.Tensor:
            flat_grad_tracking_params = __flat_params
            grad_tracking_params_containers: Tuple[
                Dict[str, Optional[torch.Tensor]], ...
            ] = pytree.tree_unflatten(  # type: ignore[assignment]
                params_containers_treespec, flat_grad_tracking_params
            )

            with container_context(params_containers, grad_tracking_params_containers):
                return objective(self, *input, **kwargs)

        objective_grad_fn = functorch.grad(objective_fn, argnums=0)
        flat_grads = objective_grad_fn(flat_params, *input, **kwargs)
        return flat_grads

    return optimality


def enable_implicit_gradients(
    cls: Type['ImplicitMetaGradientModule'],
) -> Type['ImplicitMetaGradientModule']:
    """Enables implicit gradients for the :func:`solve` method."""
    cls_solve = cls.solve
    if getattr(cls_solve, '__implicit_gradients_enabled__', False):
        raise TypeError('Implicit gradients are already enabled for the `solve` method.')

    if cls.linear_solve is not None:
        solve_kwargs = dict(solve=cls.linear_solve)
    else:
        solve_kwargs = {}

    @functools.wraps(cls_solve)
    def wrapped(  # pylint: disable=too-many-locals
        self: 'ImplicitMetaGradientModule', *input, **kwargs
    ) -> Any:
        """Solve the optimization problem."""
        params_containers = extract_module_containers(self, with_buffers=False)[0]
        meta_params_containers = [self._meta_parameters]  # pylint: disable=protected-access
        for meta_module in self.meta_children():
            meta_params_containers.extend(
                extract_module_containers(meta_module, with_buffers=False)[0]
            )
        meta_params_containers = tuple(meta_params_containers)

        flat_params: TupleOfTensors
        flat_meta_params: TupleOfTensors
        flat_params, params_containers_treespec = pytree.tree_flatten_as_tuple(
            params_containers  # type: ignore[arg-type]
        )
        flat_meta_params, meta_params_containers_treespec = pytree.tree_flatten_as_tuple(
            meta_params_containers  # type: ignore[arg-type]
        )

        def optimality_fn(
            __flat_params: TupleOfTensors,
            __flat_meta_params: TupleOfTensors,
            *input,
            **kwargs,
        ) -> TupleOfTensors:
            flat_grad_tracking_params = __flat_params
            grad_tracking_params_containers: Tuple[
                Dict[str, Optional[torch.Tensor]], ...
            ] = pytree.tree_unflatten(  # type: ignore[assignment]
                params_containers_treespec, flat_grad_tracking_params
            )
            flat_grad_tracking_meta_params = __flat_meta_params
            grad_tracking_meta_params_containers: Tuple[
                Dict[str, Optional[torch.Tensor]], ...
            ] = pytree.tree_unflatten(  # type: ignore[assignment]
                meta_params_containers_treespec, flat_grad_tracking_meta_params
            )

            with container_context(
                itertools.chain(
                    params_containers,
                    meta_params_containers,
                ),
                itertools.chain(
                    grad_tracking_params_containers,
                    grad_tracking_meta_params_containers,
                ),
            ):
                return self.optimality(*input, **kwargs)

        @custom_root(optimality_fn, argnums=1, has_aux=True, **solve_kwargs)
        def solver_fn(
            __flat_params: TupleOfTensors,  # pylint: disable=unused-argument
            __flat_meta_params: TupleOfTensors,  # pylint: disable=unused-argument
            *input,
            **kwargs,
        ) -> Tuple[TupleOfTensors, Any]:
            output = cls_solve(self, *input, **kwargs)
            flat_optimal_params: TupleOfTensors = tuple(pytree.tree_leaves(params_containers))  # type: ignore[arg-type]
            return flat_optimal_params, output

        # pylint: disable-next=unused-variable
        flat_optimal_params, output = solver_fn(flat_params, flat_meta_params, *input, **kwargs)
        return output

    wrapped.__implicit_gradients_enabled__ = True  # type: ignore[attr-defined]
    cls.solve = wrapped  # type: ignore[assignment]
    return cls


class ImplicitMetaGradientModule(MetaGradientModule):
    """The base class for differentiable implicit meta-gradient models."""

    _custom_optimality: bool
    _custom_objective: bool
    linear_solve: Optional[LinearSolver]

    def __init_subclass__(cls, linear_solve: Optional[LinearSolver] = None) -> None:
        """Validates and initializes the subclass."""
        super().__init_subclass__()
        cls.linear_solve = linear_solve

        optimality = getattr(cls, 'optimality', ImplicitMetaGradientModule.optimality)
        objective = getattr(cls, 'objective', ImplicitMetaGradientModule.objective)
        cls._custom_optimality = optimality is not ImplicitMetaGradientModule.optimality
        cls._custom_objective = objective is not ImplicitMetaGradientModule.objective

        if cls._custom_optimality:
            if isinstance(optimality, staticmethod):
                raise TypeError('method optimality() must not be a staticmethod.')
            if isinstance(optimality, classmethod):
                raise TypeError('method optimality() must not be a classmethod.')
            if not callable(optimality):
                raise TypeError('method optimality() must be callable.')
        elif not cls._custom_objective:
            raise TypeError(
                'ImplicitMetaGradientModule requires either an optimality() method or an objective() method'
            )
        else:
            if isinstance(objective, staticmethod):
                raise TypeError('method objective() must not be a staticmethod.')
            if isinstance(objective, classmethod):
                raise TypeError('method objective() must not be a classmethod.')
            if not callable(objective):
                raise TypeError('method objective() must be callable.')

            cls.optimality = make_optimality_from_objective(objective)  # type: ignore[assignment]

        enable_implicit_gradients(cls)

    def solve(self, *input, **kwargs) -> Any:
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
        """
        raise NotImplementedError  # update parameters

    def optimality(self, *input, **kwargs) -> TensorTree:
        r"""Computes the optimality residual.

        This method stands for the optimality residual to the optimal parameters after solving the
        inner optimization problem (:meth:`solve`), i.e.:

        .. code-block:: python

            module.solve(*input, **kwargs)
            module.optimality(*input, **kwargs)  # -> 0

        1. For gradient-based optimization, the :meth:`optimality` function is the KKT condition,
        usually it is the gradients of the :meth:`objective` function with respect to the module
        parameters (not the meta-parameters). If this method is not implemented, it will be
        automatically derived from the gradient of the :meth:`objective` function.

        .. math::

            \text{optimality residual} = \nabla_{\boldsymbol{x}} f (\boldsymbol{x}, \boldsymbol{\theta}) \to \boldsymbol{0}

        where :math:`\boldsymbol{x}` is the joint vector of the module parameters and
        :math:`\boldsymbol{\theta}` is the joint vector of the meta-parameters.

        References:
            - Karush-Kuhn-Tucker (KKT) conditions: https://en.wikipedia.org/wiki/Karush-Kuhn-Tucker_conditions

        2. For fixed point iteration, the :meth:`optimality` function can be the residual of the
        parameters between iterations, i.e.:

        .. math::

            \text{optimality residual} = f (\boldsymbol{x}, \boldsymbol{\theta}) - \boldsymbol{x} \to \boldsymbol{0}

        where :math:`\boldsymbol{x}` is the joint vector of the module parameters and
        :math:`\boldsymbol{\theta}` is the joint vector of the meta-parameters.

        Returns:
            A tree of tensors, the optimality residual to the optimal parameters after solving the
            inner optimization problem.
        """  # pylint: disable=line-too-long
        raise NotImplementedError

    def objective(self, *input, **kwargs) -> torch.Tensor:
        """Computes the objective function value.

        This method is used to calculate the :meth:`optimality` if it is not implemented.
        Otherwise, this method is optional.

        Returns:
            A scalar tensor (``dim=0``), the objective function value.
        """
        raise NotImplementedError
