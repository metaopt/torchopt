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

from typing import List

import torch

import torchopt
from torchopt._src import base


# from torchopt._src.base import EmptyState
# from torchopt._src.typing import ScalarOrSchedule

# mypy: ignore-errors
class FuncOptimizer:  # pylint: disable=too-few-public-methods
    """A high-level functional optimizer base class."""

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
        """Compute the gradients of loss to the network parameters and update network parameters.

        Graph of the derivative will be constructed, allowing to compute higher order derivative
        products. We use the differentiable optimizer (pass argument inplace=False) to scale the
        gradients and update the network parameters without modifying tensors in-place.

        Args:
          loss (torch.Tensor): loss that is used to compute the gradients to network parameters.
        """
        if self.optim_state is None:
            self.optim_state = self.impl.init(params)

        # step parameter only
        grad = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        updates, self.optim_state = self.impl.update(grad, self.optim_state, False)
        new_params = torchopt.apply_updates(list(params), list(updates), inplace=False)
        return new_params
