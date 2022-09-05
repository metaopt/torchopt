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

import torch
import torch.nn as nn

from torchopt._src.base import GradientTransformation
from torchopt._src.update import apply_updates
from torchopt._src.utils import pytree


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
        self.impl = impl
        self.param_containers_groups = []  # type: ignore
        self.state_groups = []  # type: ignore

        self.add_param_group(net)

    def step(self, loss: torch.Tensor):
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
            flattened_params, container_treedef = pytree.tree_flatten(param_container)
            flattened_params = tuple(flattened_params)
            grads = torch.autograd.grad(
                loss, flattened_params, create_graph=True, allow_unused=True
            )
            updates, new_state = self.impl.update(
                grads,
                new_state,
                params=flattened_params,
                inplace=False,
            )
            self.state_groups[i] = new_state
            flattened_new_params = apply_updates(flattened_params, updates, inplace=False)
            new_params = pytree.tree_unflatten(container_treedef, flattened_new_params)
            for container, new_param in zip(param_container, new_params):
                container.update(new_param)

    def add_param_group(self, net):
        """Add a param group to the optimizer's :attr:`state_groups`."""
        # pylint: disable-next=import-outside-toplevel,cyclic-import
        from torchopt._src.utils import _extract_container

        net_container = _extract_container(net, with_buffer=False)
        flattened_params = tuple(pytree.tree_leaves(net_container))
        optimizer_state = self.impl.init(flattened_params)
        self.param_containers_groups.append(net_container)
        self.state_groups.append(optimizer_state)

    def state_dict(self):
        """Extract the references of the optimizer states.

        Note that the states are references, so any in-place operations will change the states
        inside :class:`MetaOptimizer` at the same time.
        """
        return tuple(self.state_groups)

    def load_state_dict(self, state_dict):
        """Load the references of the optimizer states."""
        self.state_groups[:] = list(state_dict)
