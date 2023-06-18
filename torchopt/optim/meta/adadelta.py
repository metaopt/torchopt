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
"""Differentiable Adadelta optimizer."""

from __future__ import annotations

import torch.nn as nn

from torchopt import alias
from torchopt.optim.meta.base import MetaOptimizer
from torchopt.typing import ScalarOrSchedule


__all__ = ['MetaAdadelta']


class MetaAdadelta(MetaOptimizer):
    """The differentiable Adadelta optimizer.

    See Also:
        - The functional Adadelta optimizer: :func:`torchopt.adadetla`.
        - The classic Adadelta optimizer: :class:`torchopt.Adadelta`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        module: nn.Module,
        lr: ScalarOrSchedule = 1e-3,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        *,
        moment_requires_grad: bool = True,
        maximize: bool = False,
    ) -> None:
        """Initialize the meta-Adadelta optimizer.

        Args:
            lr (float or callable, optional): This is a fixed global scaling factor or a learning rate
                scheduler. (default: :const:`1e-3`)
            rho (float, optional): Coefficients used for computing running averages of  gradient and its square. (default: :const:`0.9`)
            eps (float, optional): A small constant applied to the square root (as in the Adadelta paper) to avoid dividing by zero when rescaling.
                (default: :const:`1e-6`)
            weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
                (default: :const:`0.0`)
            maximize (bool, optional): Maximize the params based on the objective, instead of minimizing.
                (default: :data:`False`)
        """
        super().__init__(
            module,
            alias.adadelta(
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                moment_requires_grad=moment_requires_grad,
                maximize=maximize,
            ),
        )
