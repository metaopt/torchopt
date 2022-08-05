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

from typing import Any, Callable, Optional, Union

import torch.nn as nn

from torchopt._src import base  # pylint: disable=unused-import
from torchopt._src.alias import adamw
from torchopt._src.optimizer.meta.base import MetaOptimizer
from torchopt._src.typing import ScalarOrSchedule


class MetaAdamW(MetaOptimizer):
    """The classic RMSProp optimizer.

    See Also:
        - The functional RMSProp optimizer: :func:`torchopt.rmsprop`.
        - The differentiable meta-RMSProp optimizer: :class:`torchopt.MetaRMSProp`.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        net: nn.Module,
        lr: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        moment_requires_grad: bool = False,
        use_accelerated_op: bool = False,
        weight_decay: float = 0.01,
        mask: Optional[Union[Any, Callable[['base.Params'], Any]]] = None,
    ):
        r"""The `init` function.

        Args:
            net (nn.Module): A network whose parameters should be optimized.
            lr: this is a fixed global scaling factor.
            b1: the exponential decay rate to track the first moment of past gradients.
            b2: the exponential decay rate to track the second moment of past gradients.
            eps: A small constant applied to denominator outside of the square root (as in the Adam
                paper) to avoid dividing by zero when rescaling.
            eps_root: (default: :data:`0.0`)
                A small constant applied to denominator inside the square root (as in RMSProp),
                to avoid dividing by zero when rescaling. This is needed for example when computing
                (meta-)gradients through Adam.
            moment_requires_grad: (default: :data:`False`)
                If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
                flag is often used in Meta Learning algorithms.
            use_accelerated_op: (default: :data:`False`)
                If :data:`True` use our implemented fused operator.
            weight_decay: strength of the weight decay regularization. Note that this
                weight decay is multiplied with the learning rate. This is consistent
                with other frameworks such as PyTorch, but different from
                (Loshchilov et al, 2019) where the weight decay is only multiplied with
                the "schedule multiplier", but not the base learning rate.
            mask: a tree with same structure as (or a prefix of) the params PyTree,
                or a Callable that returns such a pytree given the params/updates.
                The leaves should be booleans, `True` for leaves/subtrees you want to
                apply the weight decay to, and `False` for those you want to skip. Note
                that the Adam gradient transformations are applied to all parameters.
        """
        super().__init__(
            net,
            adamw(
                lr=lr,
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                moment_requires_grad=moment_requires_grad,
                use_accelerated_op=use_accelerated_op,
                weight_decay=weight_decay,
                mask=mask,
            ),
        )
