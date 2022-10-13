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
"""Adam optimizer."""

from typing import Iterable, Tuple

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
        lr: ScalarOrSchedule,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        eps_root: float = 0.0,
        maximize: bool = False,
        use_accelerated_op: bool = False,
    ) -> None:
        r"""The :meth:`init` function.

        Args:
            params: (iterable of torch.Tensor)
                An iterable of :class:`torch.Tensor`\s. Specifies what tensors should be optimized.
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
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
            use_accelerated_op: (default: :data:`False`)
                If :data:`True` use our implemented fused operator.
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
