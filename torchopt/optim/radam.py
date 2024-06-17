# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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
"""RAdam optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from torchopt import alias
from torchopt.optim.base import Optimizer


if TYPE_CHECKING:
    import torch

    from torchopt.typing import ScalarOrSchedule


__all__ = ['RAdam']


class RAdam(Optimizer):
    """The classic RAdam optimizer.

    See Also:
        - The functional Adam optimizer: :func:`torchopt.radam`.
        - The differentiable meta-RAdam optimizer: :class:`torchopt.MetaRAdam`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: ScalarOrSchedule = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        r"""Initialize the RAdam optimizer.

        Args:
            params (iterable of Tensor): An iterable of :class:`torch.Tensor`\s. Specifies what
                tensors should be optimized.
            lr (float or callable, optional): This is a fixed global scaling factor or a learning rate
                scheduler. (default: :const:`1e-3`)
            betas (tuple of float, optional): Coefficients used for computing running averages of
                gradient and its square. (default: :const:`(0.9, 0.999)`)
            eps (float, optional): A small constant applied to the square root (as in the RAdam paper)
                to avoid dividing by zero when rescaling.
                (default: :const:`1e-6`)
            weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
                (default: :const:`0.0`)
        """
        super().__init__(
            params,
            alias.radam(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                moment_requires_grad=False,
            ),
        )
