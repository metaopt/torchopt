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
"""The base class for optimizers."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch

from torchopt import pytree
from torchopt.base import UninitializedState
from torchopt.typing import GradientTransformation, OptState, Params, TupleOfTensors
from torchopt.update import apply_updates


__all__ = ['Optimizer']


class Optimizer:
    """A base class for classic optimizers that similar to :class:`torch.optim.Optimizer`."""

    def __init__(self, params: Iterable[torch.Tensor], impl: GradientTransformation) -> None:
        r"""Initialize the optimizer.

        Args:
            params (iterable of torch.Tensor): An iterable of :class:`torch.Tensor`\s. Specifies
                what tensors should be optimized.
            impl (GradientTransformation): A low level optimizer function, it could be a optimizer
                function provided in :mod:`torchopt.alias` or a customized :func:`torchopt.chain`\ed
                transformation.
                Note that using ``Optimizer(sgd())`` or ``Optimizer(chain(sgd()))`` is equivalent to
                :class:`torchopt.SGD`.
        """
        if not isinstance(impl, GradientTransformation):
            raise TypeError(f'{impl} (type: {type(impl).__name__}) is not a GradientTransformation')

        self.impl: GradientTransformation = impl
        self.param_groups: list[TupleOfTensors] = []
        self.param_treespecs: list[pytree.PyTreeSpec] = []
        self.state_groups: list[OptState] = []

        if not isinstance(params, (list, tuple)):
            params = tuple(params)
        self.add_param_group(params)

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""Set the gradients of all optimized :class:`torch.Tensor`\s to zero.

        The behavior is similar to :meth:`torch.optim.Optimizer.zero_grad`.

        Args:
            set_to_none (bool, optional): Instead of setting to zero, set the ``grads`` to
                :data:`None`. (default: :data:`False`)
        """
        if set_to_none:

            def f(p: torch.Tensor) -> None:
                p.grad = None

        else:

            def f(p: torch.Tensor) -> None:
                if p.grad is None:
                    return
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

        pytree.tree_map_(f, self.param_groups)  # type: ignore[arg-type]

    def state_dict(self) -> tuple[OptState, ...]:
        """Return the state of the optimizer."""
        return tuple(self.state_groups)

    def load_state_dict(self, state_dict: Sequence[OptState]) -> None:
        """Load the optimizer state.

        Args:
            state_dict (sequence of tree of Tensor): Optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.state_groups[:] = list(state_dict)

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Perform a single optimization step.

        The behavior is similar to :meth:`torch.optim.Optimizer.step`.

        Args:
            closure (callable or None, optional): A closure that reevaluates the model and returns
                the loss. Optional for most optimizers. (default: :data:`None`)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        def f(p: torch.Tensor) -> torch.Tensor | None:
            return p.grad

        for i, (params, state) in enumerate(zip(self.param_groups, self.state_groups)):
            if isinstance(state, UninitializedState):
                state = self.impl.init(params)
            grads = pytree.tree_map(f, params)  # type: ignore[arg-type]
            updates, new_state = self.impl.update(grads, state, params=params, inplace=True)
            self.param_groups[i] = apply_updates(params, updates, inplace=True)
            self.state_groups[i] = new_state

        return loss

    def add_param_group(self, params: Params) -> None:
        """Add a param group to the optimizer's ``param_groups``."""
        flat_params: TupleOfTensors
        flat_params, params_treespec = pytree.tree_flatten_as_tuple(params)
        self.param_groups.append(flat_params)
        self.param_treespecs.append(params_treespec)
        self.state_groups.append(UninitializedState())
