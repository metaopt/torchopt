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
from torch import nn
import jax

import TorchOpt
from TorchOpt._src.alias import sgd, adam
from TorchOpt._src import base
from TorchOpt._src.pytypes import ScalarOrSchedule


class MetaOptimizer(object):
    """A high-level optimizer base class for meta learning."""

    def __init__(self, net: nn.Module, impl: base.GradientTransformation):
        """
        Args:
          net (nn.Module): a network whose parameters should be optimized.
          impl (base.GradientTransformation): a low level optimizer function, it could be a 
            optimizer function provided by `alias.py` or a customerized `chain` provided by 
            `combine.py`. Note that use `MetaOptimizer(sgd(moment_requires_grad=True))` or 
            `MetaOptimizer(chain(sgd(moment_requires_grad=True))) is equavalent to `MetaSGD`.
        """
        self.impl = impl
        self.param_containers_groups = []
        self.state_groups = []
        self.add_param_group(net)

    def step(self, loss: torch.Tensor):
        """Compute the gradients of the loss to the network parameters and update network parameters.

        Graph of the derivative will be constructed, allowing to compute higher order derivative products.
        We use the differentiable optimizer (pass argument inplace=False) to scale the gradients and update
        the network parameters without modifying tensors in-place.

        Args:
          loss (torch.Tensor): the loss that is used to compute the gradients to the network parameters.
        """
        # step parameter only
        for idx, (state, param_containers) in enumerate(zip(self.state_groups, self.param_containers_groups)):
            flatten_params, containers_tree = jax.tree_util.tree_flatten(
                param_containers)
            flatten_params = tuple(flatten_params)
            grad = torch.autograd.grad(
                loss, flatten_params, create_graph=True, allow_unused=True)
            updates, state = self.impl.update(grad, state, False)
            self.state_groups[idx] = state
            new_params = TorchOpt.apply_updates(
                flatten_params, updates, inplace=False)
            unflatten_new_params = containers_tree.unflatten(new_params)
            for (container, unflatten_param) in zip(param_containers, unflatten_new_params):
                container.update(unflatten_param)

    def add_param_group(self, net):
        from .utils import _extract_container
        net_container = _extract_container(net, with_buffer=False)
        flatten_param, _ = jax.tree_util.tree_flatten(net_container)
        flatten_param = tuple(flatten_param)
        optim_state = self.impl.init(flatten_param)
        self.state_groups.append(optim_state)
        self.param_containers_groups.append(net_container)

    def state_dict(self):
        """Extract the references of the optimizer states.

        Note that the states are references, so any in-place operations will
        change the states inside `MetaOptimizer` at the same time.
        """
        out_groups = tuple(group for group in self.state_groups)
        return out_groups

    def load_state_dict(self, state_dict):
        self.state_groups = list(group for group in state_dict)


class MetaSGD(MetaOptimizer):
    """A canonical Stochastic Gradient Descent optimiser."""

    def __init__(self,
                 net,
                 lr: ScalarOrSchedule,
                 momentum: float = None,
                 nesterov: bool = False,
                 moment_requires_grad: bool = True):
        """
        Args:
          net (nn.Module): a network whose parameters should be optimized.
          args: other arguments see `alias.sgd`, here we set `moment_requires_grad=True`
            to make tensors like momentum be differentiable.
        """
        super().__init__(net,
                         sgd(lr=lr,
                             momentum=momentum,
                             nesterov=nesterov,
                             moment_requires_grad=moment_requires_grad)
                         )


class MetaAdam(MetaOptimizer):
    """The classic Adam optimiser."""

    def __init__(self,
                 net,
                 lr: ScalarOrSchedule,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 eps: float = 1e-8,
                 eps_root: float = 0.0,
                 moment_requires_grad: bool = True,
                 use_accelerated_op: bool = False):
        """
        Args:
          net (nn.Module): a network whose parameters should be optimized.
          args: other arguments see `alias.adam`, here we set `moment_requires_grad=True`
            to make tensors like momentum be differentiable.
        """
        super().__init__(net,
                         adam(lr=lr,
                              b1=b1,
                              b2=b2,
                              eps=eps,
                              eps_root=eps_root,
                              moment_requires_grad=moment_requires_grad,
                              use_accelerated_op=use_accelerated_op)
                         )
