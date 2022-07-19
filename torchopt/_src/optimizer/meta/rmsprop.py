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

from typing import Optional

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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: nn.Module,
        lr: ScalarOrSchedule,
        decay: float = 0.9,
        eps: float = 1e-8,
        initial_scale: float = 0.0,
        centered: bool = False,
        momentum: Optional[float] = None,
        nesterov: bool = False,
    ):
        """The :meth:`init` function.

        Args:
            net (nn.Module): A network whose parameters should be optimized.
            lr: This is a fixed global scaling factor.
            decay: The decay used to track the magnitude of previous gradients.
            eps: A small numerical constant to avoid dividing by zero when rescaling.
            initial_scale: (default: :data:`0.0`)
                Initialization of accumulators tracking the magnitude of previous updates. PyTorch
                uses :data:`0.0`, TensorFlow 1.x uses :data:`1.0`. When reproducing results from a
                paper, verify the value used by the authors.
            centered: (default: :data:`False`)
                Whether the second moment or the variance of the past gradients is
                used to rescale the latest gradients.
            momentum: (default: :data:`None`)
                Here we set ``moment_requires_grad=True`` to make tensors like momentum be
                differentiable.
            nesterov: (default: :data:`False`)
                Whether the nesterov momentum is used.
        """
        super().__init__(
            net,
            rmsprop(
                lr=lr,
                decay=decay,
                eps=eps,
                initial_scale=initial_scale,
                centered=centered,
                momentum=momentum,
                nesterov=nesterov,
            ),
        )
