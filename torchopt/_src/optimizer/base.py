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

from typing import Iterable

import torch

from torchopt._src.base import GradientTransformation
from torchopt._src.update import apply_updates
from torchopt._src.utils import pytree


class Optimizer:
    """A base class for classic optimizers that similar to :class:`torch.optim.Optimizer`."""

    def __init__(self, params: Iterable[torch.Tensor], impl: GradientTransformation):
        r"""The :meth:`init` function.

        Args:
            params (iterable of torch.Tensor): An iterable of :class:`torch.Tensor`\s. Specifies
                what tensors should be optimized.
            impl (GradientTransformation): A low level optimizer function, it could be a optimizer
                function provided by ``alias.py`` or a customized ``chain`` provided by
                ``combine.py``.
                Note that using ``Optimizer(sgd())`` or ``Optimizer(chain(sgd()))`` is equivalent to
                :class:`torchopt.SGD`.
        """
        self.impl = impl
        self.param_groups = []  # type: ignore
        self.param_tree_groups = []  # type: ignore
        self.state_groups = []  # type: ignore

        if not isinstance(params, list):
            params = list(params)
        self.add_param_group(params)

    def zero_grad(self, set_to_none: bool = False):
        r"""Sets the gradients of all optimized :class:`torch.Tensor`\s to zero.

        The behavior is similar to :meth:`torch.optim.Optimizer.zero_grad`.

        Args:
            set_to_none (bool): Instead of setting to zero, set the ``grads`` to :data:`None`.
        """
        for group in self.param_groups:
            if set_to_none:

                def f(p):
                    p.grad = None

            else:

                def f(p):
                    if p.grad is None:
                        return
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

            pytree.tree_map(f, group)

    def state_dict(self):
        """Returns the state of the optimizer."""
        return self.state_groups

    def load_state_dict(self, state_dict):
        """Loads the optimizer state.

        Args:
            state_dict (dict): Optimizer state. Should be an object returned from a call to
                :meth:`state_dict`.
        """
        self.state_groups = state_dict

    def step(self, closure=None):
        """Performs a single optimization step.

        The behavior is similar to :meth:`torch.optim.Optimizer.step`.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        def f(p):
            return p.grad

        for i, (params, state) in enumerate(zip(self.param_groups, self.state_groups)):
            grads = pytree.tree_map(f, params)
            updates, new_state = self.impl.update(grads, state, params=params, inplace=True)
            self.param_groups[i] = apply_updates(params, updates, inplace=True)
            self.state_groups[i] = new_state

        return loss

    def add_param_group(self, params):
        """Add a param group to the optimizer's :attr:`param_groups`."""
        params, params_tree = pytree.tree_flatten(params)
        params = tuple(params)
        self.param_groups.append(params)
        self.param_tree_groups.append(params_tree)
        self.state_groups.append(self.impl.init(params))
