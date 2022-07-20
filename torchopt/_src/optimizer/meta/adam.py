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

import torch.nn as nn

from torchopt._src.alias import adam
from torchopt._src.optimizer.meta.base import MetaOptimizer
from torchopt._src.typing import ScalarOrSchedule


class MetaAdam(MetaOptimizer):
    """The differentiable Adam optimizer.

    See Also:
        - The functional Adam optimizer: :func:`torchopt.adam`.
        - The classic Adam optimizer: :class:`torchopt.Adam`.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: nn.Module,
        lr: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        moment_requires_grad: bool = True,
        use_accelerated_op: bool = False,
    ):
        """The :meth:`init` function.

        Args:
            net (nn.Module): A network whose parameters should be optimized.
            args: Other arguments see also :func:`torchopt.adam`,
            lr: This is a fixed global scaling factor.
            b1: The exponential decay rate to track the first moment of past gradients.
            b2: The exponential decay rate to track the second moment of past gradients.
            eps: A small constant applied to denominator outside of the square root (as in the Adam
                paper) to avoid dividing by zero when rescaling.
            eps_root: (default: :data:`0.0`)
                A small constant applied to denominator inside the square root (as in RMSProp), to
                avoid dividing by zero when rescaling. This is needed for example when computing
                (meta-)gradients through Adam.
            moment_requires_grad: (default: :data:`True`)
                Here we set ``moment_requires_grad=True`` to make tensors like momentum be
                differentiable.
            use_accelerated_op: (default: :data:`False`)
                If :data:`True` use our implemented fused operator.
        """
        super().__init__(
            net,
            adam(
                lr=lr,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                moment_requires_grad=moment_requires_grad,
                use_accelerated_op=use_accelerated_op,
            ),
        )
