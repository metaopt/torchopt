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
from torchopt._src.optimizer.meta.base import MetaOptimizer
from torchopt._src.typing import ScalarOrSchedule


class MetaRMSProp(MetaOptimizer):
    """The classic RMSProp optimizer."""

    def __init__(
        self,
        net,
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
            net (nn.Module):
                A network whose parameters should be optimized.
            args:
                Other arguments see `alias.adam`, here we set `moment_requires_grad=True`
                to make tensors like momentum be differentiable.
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
                nesterov=nesterov
            )
        )
