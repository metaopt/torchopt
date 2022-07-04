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

from torchopt._src.alias import sgd
from torchopt._src.optimizer.base import Optimizer
from torchopt._src.typing import ScalarOrSchedule


class SGD(Optimizer):
    """The classic SGD optimizer."""

    def __init__(
        self,
        params,
        lr: ScalarOrSchedule,
        momentum: Union[float, None] = None,
        nesterov: bool = False
    ):
        """The `init` function.

        Args:
            params (iterable):
                An iterable of `torch.Tensor`s. Specifies what Tensors should be
                optimized.
            args:
                Other arguments see `alias.adam`.
        """

        super().__init__(
            params, sgd(lr=lr, momentum=momentum, nesterov=nesterov, moment_requires_grad=False)
        )
