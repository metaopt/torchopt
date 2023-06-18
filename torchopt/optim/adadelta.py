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
"""Adadelta optimizer."""

from __future__ import annotations

from typing import Iterable

import torch

from torchopt import alias
from torchopt.optim.base import Optimizer
from torchopt.typing import ScalarOrSchedule


__all__ = ['Adadelta']


class Adadelta(Optimizer):
    """The classic Adam optimizer.

    See Also:
        - The functional Adam optimizer: :func:`torchopt.adadelta`.
        - The differentiable meta-Adam optimizer: :class:`torchopt.MetaAdadetla`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: ScalarOrSchedule = 1e-3,
        rho: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        maximize: bool = False,
    ) -> None:
        r"""Initialize the Adadelta optimizer.

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
            params,
            alias.adadelta(
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                moment_requires_grad=False,
                maximize=maximize,
            ),
        )
