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
"""The base class for differentiable implicit meta-gradient models."""

# pylint: disable=redefined-builtin

from __future__ import annotations

import abc
import functools
import inspect
import itertools
from typing import Any, Iterable

import functorch
import torch

from torchopt.diff.implicit.decorator import custom_root
from torchopt.nn.module import MetaGradientModule
from torchopt.nn.stateless import reparametrize, swap_state
from torchopt.typing import LinearSolver, TupleOfTensors


__all__ = ['ImplicitMetaGradientModule']


def _stateless_objective_fn(
    flat_params: TupleOfTensors,
    flat_meta_params: TupleOfTensors,
    params_names: Iterable[str],
    meta_params_names: Iterable[str],
    self: ImplicitMetaGradientModule,
    /,
    *input: Any,
    **kwargs: Any,
) -> torch.Tensor:
    with reparametrize(
        self,
        itertools.chain(
            zip(params_names, flat_params),
            zip(meta_params_names, flat_meta_params),
        ),
    ):
        return self.objective(*input, **kwargs)


def _stateless_optimality_fn(
    flat_params: TupleOfTensors,
    flat_meta_params: TupleOfTensors,
    params_names: Iterable[str],
    meta_params_names: Iterable[str],
    self: ImplicitMetaGradientModule,
    /,
    *input: Any,
    **kwargs: Any,
) -> TupleOfTensors:
    with reparametrize(
        self,
        itertools.chain(
            zip(params_names, flat_params),
            zip(meta_params_names, flat_meta_params),
        ),
    ):
        return self.optimality(*input, **kwargs)


def make_optimality_from_objective(
    cls: type[ImplicitMetaGradientModule],
) -> type[ImplicitMetaGradientModule]:
    """Derive the optimality function of the objective function."""
    static_super_objective = inspect.getattr_static(ImplicitMetaGradientModule, 'objective')
    static_cls_objective = inspect.getattr_static(cls, 'objective', static_super_objective)
    if static_cls_objective is static_super_objective:
        raise TypeError('The objective function is not defined.')

    def optimality(self: ImplicitMetaGradientModule, *input: Any, **kwargs: Any) -> TupleOfTensors:
        named_params = tuple(self.named_parameters())
        named_meta_params = tuple(self.named_meta_parameters())
        if len(named_params) == 0:
            raise RuntimeError('The module has no parameters.')
        if len(named_meta_params) == 0:
            raise RuntimeError('The module has no meta-parameters.')
        params_names, flat_params = tuple(zip(*named_params))
        meta_params_names, flat_meta_params = tuple(zip(*named_meta_params))

        objective_grad_fn = functorch.grad(_stateless_objective_fn, argnums=0)
        return objective_grad_fn(
            flat_params,
            flat_meta_params,
            params_names,
            meta_params_names,
            self,
            *input,
            **kwargs,
        )

    cls.optimality = optimality  # type: ignore[method-assign]
    return cls


def enable_implicit_gradients(
    cls: type[ImplicitMetaGradientModule],
) -> type[ImplicitMetaGradientModule]:
    """Enable implicit gradients for the :func:`solve` method."""
    cls_solve = cls.solve
    if getattr(cls_solve, '__implicit_gradients_enabled__', False):
        raise TypeError('Implicit gradients are already enabled for the `solve` method.')

    solve_kwargs = {'solve': cls.linear_solve} if cls.linear_solve is not None else {}

    @custom_root(_stateless_optimality_fn, argnums=1, has_aux=True, **solve_kwargs)
    def stateless_solver_fn(
        # pylint: disable=unused-argument
        flat_params: TupleOfTensors,
        flat_meta_params: TupleOfTensors,
        params_names: Iterable[str],
        meta_params_names: Iterable[str],
        # pylint: enable=unused-argument
        self: ImplicitMetaGradientModule,
        /,
        *input: Any,
        **kwargs: Any,
    ) -> tuple[TupleOfTensors, Any]:
        """Solve the optimization problem."""
        output = cls_solve(self, *input, **kwargs)
        flat_optimal_params = tuple(p.detach_() for p in self.parameters())
        return flat_optimal_params, output

    @functools.wraps(cls_solve)
    def wrapped(self: ImplicitMetaGradientModule, *input: Any, **kwargs: Any) -> Any:
        """Solve the optimization problem."""
        named_params = tuple(self.named_parameters())
        named_meta_params = tuple(self.named_meta_parameters())
        if len(named_params) == 0:
            raise RuntimeError('The module has no parameters.')
        if len(named_meta_params) == 0:
            raise RuntimeError('The module has no meta-parameters.')
        params_names, flat_params = tuple(zip(*named_params))
        meta_params_names, flat_meta_params = tuple(zip(*named_meta_params))

        flat_optimal_params, output = stateless_solver_fn(
            flat_params,
            flat_meta_params,
            params_names,
            meta_params_names,
            self,
            *input,
            **kwargs,
        )
        swap_state(self, zip(params_names, flat_optimal_params))
        return output

    wrapped.__implicit_gradients_enabled__ = True  # type: ignore[attr-defined]
    cls.solve = wrapped  # type: ignore[method-assign]
    return cls


class ImplicitMetaGradientModule(MetaGradientModule, metaclass=abc.ABCMeta):
    """The base class for differentiable implicit meta-gradient models."""

    _custom_optimality: bool
    _custom_objective: bool
    linear_solve: LinearSolver | None

    def __init_subclass__(cls, linear_solve: LinearSolver | None = None) -> None:
        """Validate and initialize the subclass."""
        super().__init_subclass__()
        cls.linear_solve = linear_solve

        static_super_optimality = inspect.getattr_static(ImplicitMetaGradientModule, 'optimality')
        static_super_objective = inspect.getattr_static(ImplicitMetaGradientModule, 'objective')
        static_cls_optimality = inspect.getattr_static(cls, 'optimality')
        static_cls_objective = inspect.getattr_static(cls, 'objective')
        cls._custom_optimality = static_cls_optimality is not static_super_optimality
        cls._custom_objective = static_cls_objective is not static_super_objective

        if cls._custom_optimality:
            if isinstance(static_cls_optimality, staticmethod):
                raise TypeError('method optimality() must not be a staticmethod.')
            if isinstance(static_cls_optimality, classmethod):
                raise TypeError('method optimality() must not be a classmethod.')
            if not callable(static_cls_optimality):
                raise TypeError('method optimality() must be callable.')
        elif not cls._custom_objective:
            raise TypeError(
                'ImplicitMetaGradientModule requires either an optimality() method or an objective() method',
            )
        else:
            if isinstance(static_cls_objective, staticmethod):
                raise TypeError('method objective() must not be a staticmethod.')
            if isinstance(static_cls_objective, classmethod):
                raise TypeError('method objective() must not be a classmethod.')
            if not callable(static_cls_objective):
                raise TypeError('method objective() must be callable.')

            make_optimality_from_objective(cls)

        enable_implicit_gradients(cls)

    @abc.abstractmethod
    def solve(self, *input: Any, **kwargs: Any) -> Any:
        """Solve the inner optimization problem.

        .. warning::
            For gradient-based optimization methods, the parameter inputs should be explicitly
            specified in the :func:`torch.autograd.backward` function as argument ``inputs``.
            Otherwise, if not provided, the gradient is accumulated into all the leaf Tensors
            (including the meta-parameters) that were used to compute the objective output.
            Alternatively, please use :func:`torch.autograd.grad` instead.

        Examples:
            .. code-block:: python

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

    def optimality(self, *input: Any, **kwargs: Any) -> TupleOfTensors:
        r"""Compute the optimality residual.

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
            A tuple of tensors, the optimality residual to the optimal parameters after solving the
            inner optimization problem. The returned tensors should correspond to the outputs of
            `tuple(self.parameters())`.
        """  # pylint: disable=line-too-long
        raise NotImplementedError

    def objective(self, *input: Any, **kwargs: Any) -> torch.Tensor:
        """Compute the objective function value.

        This method is used to calculate the :meth:`optimality` if it is not implemented.
        Otherwise, this method is optional.

        Returns:
            A scalar tensor (``dim=0``), the objective function value.
        """
        raise NotImplementedError
