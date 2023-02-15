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
"""RMSProp optimizer."""

from typing import Iterable

import torch

from torchopt import alias
from torchopt.optim.base import Optimizer
from torchopt.typing import ScalarOrSchedule


__all__ = ['RMSProp', 'RMSprop']


class RMSProp(Optimizer):
    """The classic RMSProp optimizer.

    See Also:
        - The functional RMSProp optimizer: :func:`torchopt.rmsprop`.
        - The differentiable meta-RMSProp optimizer: :class:`torchopt.MetaRMSProp`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: ScalarOrSchedule = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        *,
        initial_scale: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
    ) -> None:
        r"""Initialize the RMSProp optimizer.

        Args:
            params (iterable of Tensor): An iterable of :class:`torch.Tensor`\s. Specifies what
                tensors should be optimized.
            lr (float or callable, optional): This is a fixed global scaling factor or a learning
                rate scheduler. (default: :const:`1e-2`)
            alpha (float, optional): Smoothing constant, the decay used to track the magnitude of
                previous gradients. (default: :const:`0.99`)
            eps (float, optional): A small numerical constant to avoid dividing by zero when
                rescaling. (default: :const:`1e-8`)
            weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
                (default: :const:`0.0`)
            momentum (float, optional): The decay rate used by the momentum term. The momentum is
                not used when it is set to :const:`0.0`. (default: :const:`0.0`)
            centered (bool, optional): If :data:`True`, use the variance of the past gradients to
                rescale the latest gradients. (default: :data:`False`)
            initial_scale (float, optional): Initialization of accumulators tracking the magnitude
                of previous updates. PyTorch uses :data:`0.0`, TensorFlow 1.x uses :data:`1.0`. When
                reproducing results from a paper, verify the value used by the authors.
                (default: :data:`0.0`)
            nesterov (bool, optional): Whether to use Nesterov momentum. (default: :data:`False`)
            maximize (bool, optional): Maximize the params based on the objective, instead of
                minimizing. (default: :data:`False`)
        """
        super().__init__(
            params,
            alias.rmsprop(
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
                initial_scale=initial_scale,
                nesterov=nesterov,
                maximize=maximize,
            ),
        )


RMSprop = RMSProp  # alias for PyTorch compatibility
