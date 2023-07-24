# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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
"""Preset :class:`GradientTransformation` for the Adan optimizer."""

from __future__ import annotations

from torchopt.alias.utils import (
    _get_use_chain_flat,
    flip_sign_and_add_weight_decay,
    scale_by_neg_lr,
)
from torchopt.combine import chain
from torchopt.transform import scale_by_adan
from torchopt.typing import GradientTransformation, ScalarOrSchedule


__all__ = ['adan']


# pylint: disable-next=too-many-arguments
def adan(
    lr: ScalarOrSchedule = 1e-3,
    betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    max_grad_norm=0.0,
    no_prox=False,
    *,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
    maximize: bool = False,
) -> GradientTransformation:
    """Create a functional version of the adan optimizer.

    adan is an SGD variant with learning rate adaptation. The *learning rate* used for each weight
    is computed from estimates of first- and second-order moments of the gradients (using suitable
    exponential moving averages).

    References:
        - Kingma et al., 2014: https://arxiv.org/abs/1412.6980

    Args:
        lr (float or callable, optional): This is a fixed global scaling factor or a learning rate
            scheduler. (default: :const:`1e-3`)
        betas (tuple of float, optional): Coefficients used for
            first- and second-order moments. (default: :const:`(0.98, 0.92, 0.99)`)
        eps (float, optional): Term added to the denominator to improve numerical stability.
            (default: :const:`1e-8`)
        eps_root (float, optional): Term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
            (default: :const:`0.0`)
        weight_decay (float, optional): Weight decay (L2 penalty).
            (default: :const:`0.0`)
        max_grad_norm (float, optional): Max norm of the gradients.
            (default: :const:`0.0`)
        no_prox (bool, optional): If :data:`True`, the proximal term is not applied.
            (default: :data:`False`)
        eps_root (float, optional): A small constant applied to denominator inside the square root
            (as in RMSProp), to avoid dividing by zero when rescaling. This is needed for example
            when computing (meta-)gradients through Adam. (default: :const:`0.0`)
        moment_requires_grad (bool, optional): If :data:`True`, states will be created with flag
            ``requires_grad = True``. (default: :data:`False`)
        maximize (bool, optional): Maximize the params based on the objective, instead of minimizing.
            (default: :data:`False`)

    Returns:
        The corresponding :class:`GradientTransformation` instance.

    See Also:
        The functional optimizer wrapper :class:`torchopt.FuncOptimizer`.
    """
    b1, b2, b3 = betas  # pylint: disable=invalid-name
    # pylint: disable=unneeded-not
    if not 0.0 <= max_grad_norm:
        raise ValueError(f'Invalid Max grad norm: {max_grad_norm}')
    if not (callable(lr) or lr >= 0.0):  # pragma: no cover
        raise ValueError(f'Invalid learning rate: {lr}')
    if not eps >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= b1 < 1.0:  # pragma: no cover
        raise ValueError(f'Invalid beta parameter at index 0: {b1}')
    if not 0.0 <= b2 < 1.0:  # pragma: no cover
        raise ValueError(f'Invalid beta parameter at index 1: {b2}')
    if not 0.0 <= b3 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 2: {b3}')
    if not weight_decay >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    # pylint: enable=unneeded-not

    chain_fn = chain
    flip_sign_and_add_weight_decay_fn = flip_sign_and_add_weight_decay
    adan_scaler_fn = scale_by_adan if no_prox else scale_by_proximal_adan
    scale_by_neg_lr_fn = scale_by_neg_lr

    if _get_use_chain_flat():  # default behavior
        chain_fn = chain_fn.flat  # type: ignore[attr-defined]
        flip_sign_and_add_weight_decay_fn = flip_sign_and_add_weight_decay_fn.flat  # type: ignore[attr-defined]
        adan_scaler_fn = adan_scaler_fn.flat  # type: ignore[attr-defined]
        scale_by_neg_lr_fn = scale_by_neg_lr_fn.flat  # type: ignore[attr-defined]

    return chain_fn(
        flip_sign_and_add_weight_decay_fn(weight_decay=weight_decay, maximize=maximize),
        adan_scaler_fn(
            b1=b1,
            b2=b2,
            b3=b3,
            eps=eps,
            eps_root=eps_root,
            moment_requires_grad=moment_requires_grad,
        ),
        scale_by_neg_lr_fn(lr),
    )
