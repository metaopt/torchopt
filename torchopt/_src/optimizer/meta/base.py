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

import jax
import torch
import torch.nn as nn

from torchopt._src.base import GradientTransformation
from torchopt._src.update import apply_updates


class MetaOptimizer:
    """The base class for high-level differentiable optimizers."""

    def __init__(self, net: nn.Module, impl: GradientTransformation):
        """The :meth:`init` function.

        Args:
            net (torch.nn.Module): A network whose parameters should be optimized.
            impl (GradientTransformation): A low level optimizer function, it could be a optimizer
                function provided by ``alias.py`` or a customized ``chain`` provided by
                ``combine.py``.
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
            loss (torch.Tensor): The loss that is used to compute the gradients to the network
                parameters.
        """  # pylint: disable=line-too-long
        # step parameter only
        for idx, (state, param_containers) in enumerate(
            zip(self.state_groups, self.param_containers_groups)
        ):
            flatten_params, containers_tree = jax.tree_util.tree_flatten(param_containers)
            flatten_params = tuple(flatten_params)
            grad = torch.autograd.grad(loss, flatten_params, create_graph=True, allow_unused=True)
            updates, state = self.impl.update(grad, state, False)
            self.state_groups[idx] = state
            new_params = apply_updates(flatten_params, updates, inplace=False)
            unflatten_new_params = containers_tree.unflatten(new_params)
            for container, unflatten_param in zip(param_containers, unflatten_new_params):
                container.update(unflatten_param)

    def add_param_group(self, net):
        """Add a param group to the optimizer's :attr:`state_groups`."""
        # pylint: disable=import-outside-toplevel,cyclic-import
        from torchopt._src.utils import _extract_container

        net_container = _extract_container(net, with_buffer=False)
        flatten_param, _ = jax.tree_util.tree_flatten(net_container)
        flatten_param = tuple(flatten_param)
        optim_state = self.impl.init(flatten_param)
        self.state_groups.append(optim_state)
        self.param_containers_groups.append(net_container)

    def state_dict(self):
        """Extract the references of the optimizer states.

        Note that the states are references, so any in-place operations will change the states
        inside :class:`MetaOptimizer` at the same time.
        """
        out_groups = tuple(group for group in self.state_groups)
        return out_groups

    def load_state_dict(self, state_dict):
        """Load the references of the optimizer states."""
        self.state_groups = list(group for group in state_dict)
