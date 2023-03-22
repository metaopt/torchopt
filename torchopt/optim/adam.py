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
"""Adam optimizer."""

from __future__ import annotations

from typing import Iterable

import torch

from torchopt import alias
from torchopt.optim.base import Optimizer
from torchopt.typing import ScalarOrSchedule


__all__ = ['Adam']


class Adam(Optimizer):
    """The classic Adam optimizer.

    See Also:
        - The functional Adam optimizer: :func:`torchopt.adam`.
        - The differentiable meta-Adam optimizer: :class:`torchopt.MetaAdam`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: ScalarOrSchedule = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        eps_root: float = 0.0,
        maximize: bool = False,
        use_accelerated_op: bool = False,
    ) -> None:
        r"""Initialize the Adam optimizer.

        Args:
            params (iterable of Tensor): An iterable of :class:`torch.Tensor`\s. Specifies what
                tensors should be optimized.
            lr (float or callable, optional): This is a fixed global scaling factor or a learning
                rate scheduler. (default: :const:`1e-3`)
            betas (tuple of float, optional): Coefficients used for computing running averages of
                gradient and its square. (default: :const:`(0.9, 0.999)`)
            eps (float, optional): A small constant applied to denominator outside of the square
                root (as in the Adam paper) to avoid dividing by zero when rescaling.
                (default: :const:`1e-8`)
            weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
                (default: :const:`0.0`)
            eps_root (float, optional): A small constant applied to denominator inside the square
                root (as in RMSProp), to avoid dividing by zero when rescaling. This is needed for
                example when computing (meta-)gradients through Adam. (default: :const:`0.0`)
            moment_requires_grad (bool, optional): If :data:`True` the momentums will be created
                with flag ``requires_grad=True``, this flag is often used in Meta-Learning
                algorithms. (default: :data:`False`)
            maximize (bool, optional): Maximize the params based on the objective, instead of
                minimizing. (default: :data:`False`)
            use_accelerated_op (bool, optional): If :data:`True` use our implemented fused operator.
                (default: :data:`False`)
        """
        super().__init__(
            params,
            alias.adam(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                eps_root=eps_root,
                moment_requires_grad=False,
                maximize=maximize,
                use_accelerated_op=use_accelerated_op,
            ),
        )
