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

from torchopt._src.alias import adam
from torchopt._src.optimizer.base import Optimizer
from torchopt._src.typing import ScalarOrSchedule


class Adam(Optimizer):
    """A canonical Stochastic Gradient Descent optimizer."""

    def __init__(
        self,
        params,
        lr: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        use_accelerated_op: bool = False
    ):
        """The `init` function.

        Args:
            params (iterable):
                An iterable of `torch.Tensor`s. Specifies what Tensors should be
                optimized.
            args:
                Other arguments see `alias.sgd`.
        """

        super().__init__(
            params,
            adam(
                lr=lr,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                moment_requires_grad=False,
                use_accelerated_op=use_accelerated_op
            )
        )
