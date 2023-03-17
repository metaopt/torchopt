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
"""The base class for differentiable meta-optimizers."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from torchopt import pytree
from torchopt.base import UninitializedState
from torchopt.typing import GradientTransformation, ModuleTensorContainers, OptState, TupleOfTensors
from torchopt.update import apply_updates
from torchopt.utils import extract_module_containers


__all__ = ['MetaOptimizer']


class MetaOptimizer:
    """The base class for high-level differentiable optimizers."""

    def __init__(self, module: nn.Module, impl: GradientTransformation) -> None:
        r"""Initialize the meta-optimizer.

        Args:
            module (nn.Module): A network whose parameters should be optimized.
            impl (GradientTransformation): A low level optimizer function, it could be a optimizer
                function provided in :mod:`torchopt.alias` or a customized :func:`torchopt.chain`\ed
                transformation.
                Note that using ``MetaOptimizer(sgd(moment_requires_grad=True))`` or
                ``MetaOptimizer(chain(sgd(moment_requires_grad=True)))`` is equivalent to
                :class:`torchopt.MetaSGD`.
        """
        if not isinstance(impl, GradientTransformation):
            raise TypeError(f'{impl} (type: {type(impl).__name__}) is not a GradientTransformation')

        self.impl: GradientTransformation = impl
        self.param_containers_groups: list[ModuleTensorContainers] = []
        self.state_groups: list[OptState] = []

        self.add_param_group(module)

    def step(self, loss: torch.Tensor) -> None:  # pylint: disable=too-many-locals
        """Compute the gradients of the loss to the network parameters and update network parameters.

        Graph of the derivative will be constructed, allowing to compute higher order derivative
        products. We use the differentiable optimizer (pass argument ``inplace=False``) to scale the
        gradients and update the network parameters without modifying tensors in-place.

        Args:
            loss (torch.Tensor): The loss that is used to compute the gradients to the network
                parameters.
        """
        # Step parameter only
        for i, (param_container, state) in enumerate(
            zip(self.param_containers_groups, self.state_groups),
        ):
            flat_params: TupleOfTensors
            flat_params, container_treespec = pytree.tree_flatten_as_tuple(param_container)  # type: ignore[arg-type]
            if isinstance(state, UninitializedState):
                state = self.impl.init(flat_params)
            grads = torch.autograd.grad(
                loss,
                flat_params,
                create_graph=True,
                allow_unused=True,
            )
            updates, new_state = self.impl.update(
                grads,
                state,
                params=flat_params,
                inplace=False,
            )
            self.state_groups[i] = new_state
            flat_new_params = apply_updates(flat_params, updates, inplace=False)
            new_params: ModuleTensorContainers = pytree.tree_unflatten(  # type: ignore[assignment]
                container_treespec,
                flat_new_params,
            )
            for container, new_param in zip(param_container, new_params):
                container.update(new_param)

    def add_param_group(self, module: nn.Module) -> None:
        """Add a param group to the optimizer's ``state_groups``."""
        params_container = extract_module_containers(module, with_buffers=False)[0]
        self.param_containers_groups.append(params_container)
        self.state_groups.append(UninitializedState())

    def state_dict(self) -> tuple[OptState, ...]:
        """Extract the references of the optimizer states.

        Note that the states are references, so any in-place operations will change the states
        inside :class:`MetaOptimizer` at the same time.
        """
        return tuple(self.state_groups)

    def load_state_dict(self, state_dict: Sequence[OptState]) -> None:
        """Load the references of the optimizer states."""
        self.state_groups[:] = list(state_dict)
