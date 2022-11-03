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
"""Differentiable AdamW optimizer."""

from typing import Any, Callable, Optional, Tuple, Union

import torch.nn as nn

from torchopt import alias
from torchopt.optim.meta.base import MetaOptimizer
from torchopt.typing import Params, ScalarOrSchedule


__all__ = ['MetaAdamW']


class MetaAdamW(MetaOptimizer):
    """The differentiable AdamW optimizer.

    See Also:
        - The functional AdamW optimizer: :func:`torchopt.adamw`.
        - The classic AdamW optimizer: :class:`torchopt.AdamW`.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        module: nn.Module,
        lr: ScalarOrSchedule = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        eps_root: float = 0.0,
        mask: Optional[Union[Any, Callable[[Params], Any]]] = None,
        moment_requires_grad: bool = False,
        maximize: bool = False,
        use_accelerated_op: bool = False,
    ) -> None:
        """The :meth:`init` function.

        Args:
            module: (nn.Module)
                A network whose parameters should be optimized.
            lr: (default: :const:`1e-3`)
                This is a fixed global scaling factor.
            betas: (default: :const:`(0.9, 0.999)`)
                Coefficients used for computing running averages of gradient and its square.
            eps: (default: :const:`1e-8`)
                A small constant applied to denominator outside of the square root (as in the Adam
                paper) to avoid dividing by zero when rescaling.
            weight_decay: (default: :const:`1e-2`)
                Strength of the weight decay regularization. Note that this weight decay is
                multiplied with the learning rate. This is consistent with other frameworks such as
                PyTorch, but different from (Loshchilov et al, 2019) where the weight decay is only
                multiplied with the "schedule multiplier", but not the base learning rate.
            eps_root: (default: :data:`0.0`)
                A small constant applied to denominator inside the square root (as in RMSProp), to
                avoid dividing by zero when rescaling. This is needed for example when computing
                (meta-)gradients through Adam.
            mask: (default: :data:`None`)
                A tree with same structure as (or a prefix of) the params PyTree, or a Callable that
                returns such a pytree given the params/updates. The leaves should be booleans,
                :data:`True` for leaves/subtrees you want to apply the weight decay to, and
                :data:`False` for those you want to skip. Note that the Adam gradient
                transformations are applied to all parameters.
            moment_requires_grad: (default: :data:`False`)
                If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
                flag is often used in Meta-Learning algorithms.
            maximize: (default: :data:`False`)
                Maximize the params based on the objective, instead of minimizing.
            use_accelerated_op: (default: :data:`False`)
                If :data:`True` use our implemented fused operator.
        """
        super().__init__(
            module,
            alias.adamw(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                eps_root=eps_root,
                mask=mask,
                moment_requires_grad=moment_requires_grad,
                maximize=maximize,
                use_accelerated_op=use_accelerated_op,
            ),
        )
