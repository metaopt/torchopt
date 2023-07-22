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
"""AdaGrad optimizer."""

from __future__ import annotations

from typing import Iterable

import torch

from torchopt import alias
from torchopt.optim.base import Optimizer
from torchopt.typing import ScalarOrSchedule


__all__ = ['AdaGrad', 'Adagrad']


class AdaGrad(Optimizer):
    """The classic AdaGrad optimizer.

    See Also:
        - The functional AdaGrad optimizer: :func:`torchopt.adagrad`.
        - The differentiable meta-AdaGrad optimizer: :class:`torchopt.MetaAdaGrad`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: ScalarOrSchedule = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        *,
        maximize: bool = False,
    ) -> None:
        r"""Initialize the AdaGrad optimizer.

        Args:
            params (iterable of Tensor): An iterable of :class:`torch.Tensor`\s. Specifies what
                tensors should be optimized.
            lr (float or callable, optional): This is a fixed global scaling factor or a learning
                rate scheduler. (default: :const:`1e-2`)
            lr_decay (float, optional): Learning rate decay. (default: :const:`0.0`)
            weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
                (default: :const:`0.0`)
            initial_accumulator_value (float, optional): Initial value for the accumulator.
                (default: :const:`0.0`)
            eps (float, optional): A small constant applied to denominator outside of the square
                root (as in the Adam paper) to avoid dividing by zero when rescaling.
                (default: :const:`1e-10`)
            maximize (bool, optional): Maximize the params based on the objective, instead of
                minimizing. (default: :data:`False`)
        """
        super().__init__(
            params,
            alias.adagrad(
                lr=lr,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
                maximize=maximize,
            ),
        )


Adagrad = AdaGrad  # alias for PyTorch compatibility
