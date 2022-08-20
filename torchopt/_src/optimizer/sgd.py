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

from typing import Iterable, Optional

import torch

from torchopt._src.alias import sgd
from torchopt._src.optimizer.base import Optimizer
from torchopt._src.typing import ScalarOrSchedule


class SGD(Optimizer):
    """The classic SGD optimizer.

    See Also:
        - The functional SGD optimizer: :func:`torchopt.sgd`.
        - The differentiable meta-SGD optimizer: :class:`torchopt.MetaSGD`.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: ScalarOrSchedule,
        momentum: Optional[float] = None,
        nesterov: bool = False,
    ):
        r"""The :meth:`init` function.

        Args:
            params (iterable of torch.Tensor): An iterable of :class:`torch.Tensor`\s. Specifies
                what tensors should be optimized.
            lr: This is a fixed global scaling factor.
            momentum: (default: :data:`None`)
                The ``decay`` rate used by the momentum term, when it is set to :data:`None`, then
                momentum is not used at all.
            nesterov: (default: :data:`False`)
                Whether the nesterov momentum is used.
        """
        super().__init__(
            params,
            sgd(lr=lr, momentum=momentum, nesterov=nesterov, moment_requires_grad=False),
        )
