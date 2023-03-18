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
"""AdaGrad optimizer."""

from __future__ import annotations

from typing import Iterable

import torch

from torchopt import alias
from torchopt.optim.base import Optimizer
from torchopt.typing import Scalar


__all__ = ['Adagrad']


class Adagrad(Optimizer):
    """The classic AdaGrad optimizer.

    See Also:
        - The functional AdaGrad optimizer: :func:`torchopt.adagrad`.
        - The differentiable meta AdaGrad optimizer: :class:`torchopt.MetaAdagrad`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: Scalar = 1e-2,
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
            lr: (default: :const:`1e-2`)
                This is a fixed global scaling factor.
            lr_decay: (default: :const:`0.0`)
                Learning rate decay.
            weight_decay: (default: :const:`0.0`)
                Weight decay, add L2 penalty to parameters.
            initial_accumulator_value: (default: :const:`0.0`)
                Initial value for the accumulator.
            eps: (default: :const:`1e-10`)
                A small constant applied to denominator outside of the square root (as in the AdaGrad
                paper) to avoid dividing by zero when rescaling.
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
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
