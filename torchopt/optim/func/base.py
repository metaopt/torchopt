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
"""Functional optimizer wrappers."""

from __future__ import annotations

import torch

from torchopt.base import GradientTransformation, UninitializedState
from torchopt.typing import OptState, Params
from torchopt.update import apply_updates


__all__ = ['FuncOptimizer']


class FuncOptimizer:  # pylint: disable=too-few-public-methods
    """A wrapper class to hold the functional optimizer.

    This wrapper makes it easier to maintain the optimizer states. The optimizer states are held by
    the wrapper internally. The wrapper provides a :meth:`step` function to compute the gradients
    and update the parameters.

    See Also:
        - The functional AdaDelta optimizer: :func:`torchopt.adadelta`.
        - The functional AdaGrad optimizer: :func:`torchopt.adagrad`.
        - The functional Adam optimizer: :func:`torchopt.adam`.
        - The functional AdamW optimizer: :func:`torchopt.adamw`.
        - The functional AdaMax optimizer: :func:`torchopt.adamax`.
        - The functional RAdam optimizer: :func:`torchopt.radam`.
        - The functional RMSprop optimizer: :func:`torchopt.rmsprop`.
        - The functional SGD optimizer: :func:`torchopt.sgd`.
    """

    def __init__(self, impl: GradientTransformation, *, inplace: bool = False) -> None:
        r"""Initialize the functional optimizer wrapper.

        Args:
            impl (GradientTransformation): A low level optimizer function, it could be a optimizer
                function provided in :mod:`torchopt.alias` or a customized :func:`torchopt.chain`\ed
                transformation.
            inplace (bool, optional): The default value of ``inplace`` for each optimization update.
                (default: :data:`False`)
        """
        if not isinstance(impl, GradientTransformation):
            raise TypeError(f'{impl} (type: {type(impl).__name__}) is not a GradientTransformation')

        self.impl: GradientTransformation = impl
        self.optim_state: OptState | None = UninitializedState()
        self.inplace: bool = bool(inplace)

    def step(
        self,
        loss: torch.Tensor,
        params: Params,
        inplace: bool | None = None,
    ) -> Params:
        r"""Compute the gradients of loss to the network parameters and update network parameters.

        Graph of the derivative will be constructed, allowing to compute higher order derivative
        products. We use the differentiable optimizer (pass argument inplace=False) to scale the
        gradients and update the network parameters without modifying tensors in-place.

        Args:
            loss (Tensor): The loss that is used to compute the gradients to network parameters.
            params (tree of Tensor): An tree of :class:`torch.Tensor`\s. Specifies what tensors
                should be optimized.
            inplace (bool or None, optional): Whether to update the parameters in-place. If
                :data:`None`, use the default value specified in the constructor.
                (default: :data:`None`)
        """
        if isinstance(self.optim_state, UninitializedState):
            self.optim_state = self.impl.init(params)

        if inplace is None:
            inplace = self.inplace

        # Step parameter only
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        updates, self.optim_state = self.impl.update(
            grads,
            self.optim_state,
            params=params,
            inplace=inplace,
        )
        return apply_updates(params, updates, inplace=inplace)

    def state_dict(self) -> OptState:
        """Extract the references of the optimizer states.

        Note that the states are references, so any in-place operations will change the states
        inside :class:`FuncOptimizer` at the same time.
        """
        return self.optim_state

    def load_state_dict(self, state_dict: OptState) -> None:
        """Load the references of the optimizer states."""
        self.optim_state = state_dict
