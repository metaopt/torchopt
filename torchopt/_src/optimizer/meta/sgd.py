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

from torchopt._src.alias import sgd
from torchopt._src.optimizer.meta.base import MetaOptimizer
from torchopt._src.typing import ScalarOrSchedule


class MetaSGD(MetaOptimizer):
    """The differentiable Stochastic Gradient Descent optimizer.

    See Also:
        - The functional SGD optimizer: :func:`torchopt.sgd`.
        - The classic SGD optimizer: :class:`torchopt.SGD`.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: nn.Module,
        lr: ScalarOrSchedule,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        moment_requires_grad: bool = True,
    ):
        """The :meth:`init` function.

        Args:
            net: A network whose parameters should be optimized.
            lr: This is a fixed global scaling factor.
            momentum: The ``decay`` rate used by the momentum term, when it is set to :data:`None`,
                then momentum is not used at all.
            nesterov: Whether the nesterov momentum is used.
            moment_requires_grad: Here we set ``moment_requires_grad=True`` to make tensors like
                momentum be differentiable.
        """
        super().__init__(
            net,
            sgd(
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
                moment_requires_grad=moment_requires_grad,
            ),
        )
