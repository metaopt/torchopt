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
        r"""The `init` function.

        Args:
            params: (iterable of torch.Tensor)
                An iterable of :class:`torch.Tensor`\s. Specifies what Tensors should be optimized.
            lr: (default: :const:`1e-2`)
                This is a fixed global scaling factor.
            alpha: (default: :const:`0.99`)
                Smoothing constant, the decay used to track the magnitude of previous gradients.
            eps: (default: :const:`1e-8`)
                A small numerical constant to avoid dividing by zero when rescaling.
            weight_decay: (default: :const:`0.0`)
                Weight decay, add L2 penalty to parameters.
            momentum: (default: :const:`0.0`)
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
                Whether to use Nesterov momentum.
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
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
