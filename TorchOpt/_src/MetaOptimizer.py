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
from typing import List

import TorchOpt
from TorchOpt._src import base
from TorchOpt._src.alias import sgd, adam
from TorchOpt._src.base import EmptyState
from TorchOpt._src.pytypes import ScalarOrSchedule


class MetaOptimizer(object):
    """A high-level optimizer base class for meta learning."""

    def __init__(self, impl: base.GradientTransformation):
        """
        Args:
          impl (base.GradientTransformation): a low level optimizer function, it could be a
            optimizer function provided by `alias.py` or a customerized `chain` provided by 
            `combine.py`. Note that use `MetaOptimizer(sgd(moment_requires_grad=True))` or 
            `MetaOptimizer(chain(sgd(moment_requires_grad=True))) is equavalent to `MetaSGD`.
        """
        self.impl = impl
        self.optim_state = None

    def step(self, loss: torch.Tensor, params: List[torch.Tensor]):
        """Compute the gradients of the loss to the network parameters and update network parameters.

        Graph of the derivative will be constructed, allowing to compute higher order derivative products.
        We use the differentiable optimizer (pass argument inplace=False) to scale the gradients and update
        the network parameters without modifying tensors in-place.

        Args:
          loss (torch.Tensor): the loss that is used to compute the gradients to the network parameters.
        """
        if self.optim_state is None:
            self.optim_state = self.impl.init(params)

        # step parameter only
        grad = torch.autograd.grad(
            loss, params, create_graph=True, allow_unused=True)
        updates, self.optim_state = self.impl.update(grad, self.optim_state, False)
        new_params = TorchOpt.apply_updates(
            list(params), list(updates), inplace=False)
        return new_params


class MetaSGD(MetaOptimizer):
    """A canonical Stochastic Gradient Descent optimiser."""

    def __init__(self,
                 lr: ScalarOrSchedule,
                 momentum: float = None,
                 nesterov: bool = False,
                 moment_requires_grad: bool = True):
        """
        Args:
          args: other arguments see `alias.sgd`, here we set `moment_requires_grad=True`
            to make tensors like momentum be differentiable.
        """
        super().__init__(
            sgd(lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                moment_requires_grad=moment_requires_grad)
        )


class MetaAdam(MetaOptimizer):
    """The classic Adam optimiser."""

    def __init__(
        self,
        lr: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        moment_requires_grad: bool = True,
        use_accelerated_op: bool = False):
        """
        Args:
          args: other arguments see `alias.adam`, here we set `moment_requires_grad=True`
            to make tensors like momentum be differentiable.
        """
        super().__init__(
            adam(lr=lr,
                 b1=b1,
                 b2=b2,
                 eps=eps,
                 eps_root=eps_root,
                 moment_requires_grad=moment_requires_grad,
                 use_accelerated_op=use_accelerated_op)
        )
