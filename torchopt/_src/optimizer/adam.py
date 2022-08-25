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

from typing import Iterable, Tuple

import torch

from torchopt._src.alias import adam
from torchopt._src.optimizer.base import Optimizer
from torchopt._src.typing import ScalarOrSchedule


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
        *,
        eps_root: float = 0.0,
        maximize: bool = False,
        use_accelerated_op: bool = False,
    ):
        r"""The :meth:`init` function.

        Args:
            params: (iterable of torch.Tensor)
                An iterable of :class:`torch.Tensor`\s. Specifies what tensors should be optimized.
            lr: (float, default: :const:`1e-3`)
                This is a fixed global scaling factor.
            betas: (Tuple[float, float], default: :const:`1e-3`)
                Coefficients used for computing running averages of gradient and its square.
            eps: (float, default: :const:`1e-8`)
                A small constant applied to denominator outside of the square root (as in the Adam
                paper) to avoid dividing by zero when rescaling.
            eps_root: (float, default: :data:`0.0`)
                A small constant applied to denominator inside the square root (as in RMSProp), to
                avoid dividing by zero when rescaling. This is needed for example when computing
                (meta-)gradients through Adam.
            maximize: (bool, default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
            use_accelerated_op: (bool, default: :data:`False`)
                If :data:`True` use our implemented fused operator.
        """
        super().__init__(
            params,
            adam(
                lr=lr,
                betas=betas,
                eps=eps,
                eps_root=eps_root,
                moment_requires_grad=False,
                maximize=maximize,
                use_accelerated_op=use_accelerated_op,
            ),
        )
