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
"""SGD optimizer."""

from typing import Iterable

import torch

from torchopt import alias
from torchopt.optim.base import Optimizer
from torchopt.typing import ScalarOrSchedule


__all__ = ['SGD']


class SGD(Optimizer):
    """The classic SGD optimizer.

    See Also:
        - The functional SGD optimizer: :func:`torchopt.sgd`.
        - The differentiable meta-SGD optimizer: :class:`torchopt.MetaSGD`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: ScalarOrSchedule,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        r"""Initialize the SGD optimizer.

        Args:
            params (iterable of Tensor): An iterable of :class:`torch.Tensor`\s. Specifies what
                tensors should be optimized.
            lr (float or callable): This is a fixed global scaling factor or a learning rate
                scheduler.
            momentum (float, optional): The decay rate used by the momentum term. The momentum is
                not used when it is set to :const:`0.0`. (default: :const:`0.0`)
            weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
                (default: :const:`0.0`)
            dampening (float, optional): Dampening for momentum. (default: :const:`0.0`)
            nesterov (bool, optional): Whether to use Nesterov momentum. (default: :data:`False`)
            moment_requires_grad (bool, optional): If :data:`True` the momentums will be created
                with flag ``requires_grad=True``, this flag is often used in Meta-Learning
                algorithms. (default: :data:`False`)
            maximize (bool, optional): Maximize the params based on the objective, instead of
                minimizing. (default: :data:`False`)
        """
        super().__init__(
            params,
            alias.sgd(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=dampening,
                nesterov=nesterov,
                moment_requires_grad=False,
                maximize=maximize,
            ),
        )
