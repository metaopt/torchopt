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
"""Differentiable SGD optimizer."""

import torch.nn as nn

from torchopt import alias
from torchopt.optim.meta.base import MetaOptimizer
from torchopt.typing import ScalarOrSchedule


__all__ = ['MetaSGD']


class MetaSGD(MetaOptimizer):
    """The differentiable Stochastic Gradient Descent optimizer.

    See Also:
        - The functional SGD optimizer: :func:`torchopt.sgd`.
        - The classic SGD optimizer: :class:`torchopt.SGD`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        module: nn.Module,
        lr: ScalarOrSchedule,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
        moment_requires_grad: bool = True,
        maximize: bool = False,
    ) -> None:
        """The :meth:`init` function.

        Args:
            module: (nn.Module)
                A network whose parameters should be optimized.
            lr: This is a fixed global scaling factor.
            momentum: (default: :const:`0.0`)
                The decay rate used by the momentum term. The momentum is not used when it is set to
                :const:`0.0`.
            weight_decay: (default: :const:`0.0`)
                Weight decay, add L2 penalty to parameters.
            dampening: (default: :const:`0.0`)
                Dampening for momentum.
            nesterov: (default: :const:`False`)
                Whether to use Nesterov momentum.
            moment_requires_grad: (default: :data:`True`)
                If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
                flag is often used in Meta-Learning algorithms.
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
        """
        super().__init__(
            module,
            alias.sgd(
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                dampening=dampening,
                nesterov=nesterov,
                moment_requires_grad=moment_requires_grad,
                maximize=maximize,
            ),
        )
