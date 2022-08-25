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

from torchopt._src.alias import rmsprop
from torchopt._src.optimizer.meta.base import MetaOptimizer
from torchopt._src.typing import ScalarOrSchedule


class MetaRMSProp(MetaOptimizer):
    """The differentiable RMSProp optimizer.

    See Also:
        - The functional RMSProp optimizer: :func:`torchopt.rmsprop`.
        - The classic RMSProp optimizer: :class:`torchopt.RMSProp`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        net: nn.Module,
        lr: ScalarOrSchedule = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        momentum: float = 0.0,
        centered: bool = False,
        *,
        initial_scale: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ):
        """The :meth:`init` function.

        Args:
            net: (nn.Module)
                A network whose parameters should be optimized.
            lr: (float, default: :const:`1e-2`)
                This is a fixed global scaling factor.
            alpha: (float, default: :const:`0.99`)
                Smoothing constant, the decay used to track the magnitude of previous gradients.
            eps: (float, default: :const:`1e-8`)
                A small numerical constant to avoid dividing by zero when rescaling.
            momentum: (float, default: :const:`0.0`)
                The decay rate used by the momentum term. The momentum is not used when it is set to
                :const:`0.0`.
            centered: (default: :data:`False`)
                If :data:`True`, use the variance of the past gradients to rescale the latest
                gradients.
            initial_scale: (default: :data:`0.0`)
                Initialization of accumulators tracking the magnitude of previous updates. PyTorch
                uses :data:`0.0`, TensorFlow 1.x uses :data:`1.0`. When reproducing results from a
                paper, verify the value used by the authors.
            nesterov: (default: :data:`False`)
                Whether the nesterov momentum is used.
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
        """
        super().__init__(
            net,
            rmsprop(
                lr=lr,
                alpha=alpha,
                eps=eps,
                momentum=momentum,
                centered=centered,
                initial_scale=initial_scale,
                nesterov=nesterov,
                maximize=maximize,
            ),
        )
