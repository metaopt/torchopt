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
"""Differentiable Adam optimizer."""

from typing import Tuple

import torch.nn as nn

from torchopt import alias
from torchopt.optim.meta.base import MetaOptimizer
from torchopt.typing import ScalarOrSchedule


__all__ = ['MetaAdam']


class MetaAdam(MetaOptimizer):
    """The differentiable Adam optimizer.

    See Also:
        - The functional Adam optimizer: :func:`torchopt.adam`.
        - The classic Adam optimizer: :class:`torchopt.Adam`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        module: nn.Module,
        lr: ScalarOrSchedule = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        eps_root: float = 0.0,
        moment_requires_grad: bool = True,
        maximize: bool = False,
        use_accelerated_op: bool = False,
    ) -> None:
        """The :meth:`init` function.

        Args:
            module: (nn.Module)
                A network whose parameters should be optimized.
            lr: (default: :const:`1e-3`)
                This is a fixed global scaling factor.
            betas: (default: :const:`(0.9, 0.999)`)
                Coefficients used for computing running averages of gradient and its square.
            eps: (default: :const:`1e-8`)
                A small constant applied to denominator outside of the square root (as in the Adam
                paper) to avoid dividing by zero when rescaling.
            weight_decay: (default: :const:`0.0`)
                Weight decay, add L2 penalty to parameters.
            eps_root: (default: :data:`0.0`)
                A small constant applied to denominator inside the square root (as in RMSProp), to
                avoid dividing by zero when rescaling. This is needed for example when computing
                (meta-)gradients through Adam.
            moment_requires_grad: (default: :data:`True`)
                If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
                flag is often used in Meta-Learning algorithms.
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
            use_accelerated_op: (default: :data:`False`)
                If :data:`True` use our implemented fused operator.
        """
        super().__init__(
            module,
            alias.adam(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                eps_root=eps_root,
                moment_requires_grad=moment_requires_grad,
                maximize=maximize,
                use_accelerated_op=use_accelerated_op,
            ),
        )
