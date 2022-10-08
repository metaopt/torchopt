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
"""The base class for differentiable meta-optimizers."""

from typing import Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn

from torchopt import pytree
from torchopt.typing import GradientTransformation, OptState
from torchopt.update import apply_updates


__all__ = ['MetaOptimizer']


class MetaOptimizer:
    """The base class for high-level differentiable optimizers."""

    def __init__(self, net: nn.Module, impl: GradientTransformation):
        """The :meth:`init` function.

        Args:
            net: (nn.Module)
                A network whose parameters should be optimized.
            impl: (GradientTransformation)
                A low level optimizer function, it could be a optimizer function provided by
                ``alias.py`` or a customized ``chain`` provided by ``combine.py``.
                Note that using ``MetaOptimizer(sgd(moment_requires_grad=True))`` or
                ``MetaOptimizer(chain(sgd(moment_requires_grad=True)))`` is equivalent to
                :class:`torchopt.MetaSGD`.
        """
        if not isinstance(impl, GradientTransformation):
            raise TypeError(f'{impl} (type: {type(impl).__name__}) is not a GradientTransformation')

        self.impl: GradientTransformation = impl
        self.param_containers_groups: List[Tuple[Dict[str, Optional[torch.Tensor]], ...]] = []
        self.state_groups: List[OptState] = []

        self.add_param_group(net)

    def step(self, loss: torch.Tensor):  # pylint: disable=too-many-locals
        """Compute the gradients of the loss to the network parameters and update network parameters.

        Graph of the derivative will be constructed, allowing to compute higher order derivative
        products. We use the differentiable optimizer (pass argument ``inplace=False``) to scale the
        gradients and update the network parameters without modifying tensors in-place.

        Args:
            loss: (torch.Tensor)
                The loss that is used to compute the gradients to the network parameters.
        """  # pylint: disable=line-too-long
        # Step parameter only
        for i, (param_container, new_state) in enumerate(
            zip(self.param_containers_groups, self.state_groups)
        ):
            flattened_params_or_none: List[Optional[torch.Tensor]]
            flattened_params_or_none, container_treedef = pytree.tree_flatten(param_container)  # type: ignore[arg-type]
            flattened_params = cast(
                Tuple[torch.Tensor, ...],
                tuple(filter(torch.is_tensor, flattened_params_or_none)),
            )
            grads = torch.autograd.grad(
                loss,
                flattened_params,
                create_graph=True,
                allow_unused=True,
            )
            updates, new_state = self.impl.update(
                grads,
                new_state,
                params=flattened_params,
                inplace=False,
            )
            self.state_groups[i] = new_state
            flattened_new_params = apply_updates(flattened_params, updates, inplace=False)
            new_params_iter = iter(flattened_new_params)
            flattened_new_params_or_none = [
                next(new_params_iter)
                if isinstance(old_param_or_none, torch.Tensor)
                else old_param_or_none
                for old_param_or_none in flattened_params_or_none
            ]
            new_params = cast(
                Tuple[Dict[str, Optional[torch.Tensor]], ...],
                pytree.tree_unflatten(container_treedef, flattened_new_params_or_none),
            )
            for container, new_param in zip(param_container, new_params):
                container.update(new_param)

    def add_param_group(self, net: nn.Module) -> None:
        """Add a param group to the optimizer's :attr:`state_groups`."""
        # pylint: disable-next=import-outside-toplevel
        from torchopt.utils import _extract_container

        params_container, _ = _extract_container(net, with_buffers=False)
        flattened_params = tuple(
            filter(
                torch.is_tensor,  # type: ignore[arg-type]
                # pylint: disable-next=line-too-long
                cast(List[Optional[torch.Tensor]], pytree.tree_leaves(params_container)),  # type: ignore[arg-type]
            )
        )
        optimizer_state = self.impl.init(flattened_params)
        self.param_containers_groups.append(params_container)
        self.state_groups.append(optimizer_state)

    def state_dict(self) -> Tuple['OptState', ...]:
        """Extract the references of the optimizer states.

        Note that the states are references, so any in-place operations will change the states
        inside :class:`MetaOptimizer` at the same time.
        """
        return tuple(self.state_groups)

    def load_state_dict(self, state_dict: Sequence['OptState']) -> None:
        """Load the references of the optimizer states."""
        self.state_groups[:] = list(state_dict)
