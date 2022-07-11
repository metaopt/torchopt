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

from typing import Union

from torchopt._src.alias import rmsprop
from torchopt._src.optimizer.base import Optimizer
from torchopt._src.typing import ScalarOrSchedule


class RMSProp(Optimizer):
    """An RMSProp optimizer."""

    def __init__(
        self,
        params,
        lr: ScalarOrSchedule,
        decay: float = 0.9,
        eps: float = 1e-8,
        initial_scale: float = 0.,
        centered: bool = False,
        momentum: Union[float, None] = None,
        nesterov: bool = False
    ):
        """The `init` function.

        Args:
            params (iterable): An iterable of `torch.Tensor`s. Specifies what Tensors should be
                optimized.
            lr: This is a fixed global scaling factor.
            decay: The decay used to track the magnitude of previous gradients.
            eps: A small numerical constant to avoid dividing by zero when rescaling.
            initial_scale: (default `0.`)
                Initialization of accumulators tracking the magnitude of previous
                updates. PyTorch uses `0`, TF1 uses `1`. When reproducing results
                from a paper, verify the value used by the authors.
            centered: (default `False`)
                Whether the second moment or the variance of the past gradients is
                used to rescale the latest gradients.
            momentum: (default `None`)
                The `decay` rate used by the momentum term, when it is set to `None`,
                then momentum is not used at all.
            nesterov: (default `False`) Whether nesterov momentum is used.
        """
        super().__init__(
            params,
            rmsprop(
                lr=lr,
                decay=decay,
                eps=eps,
                initial_scale=initial_scale,
                centered=centered,
                momentum=momentum,
                nesterov=nesterov
            )
        )
