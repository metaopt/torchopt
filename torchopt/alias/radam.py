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
"""Preset :class:`GradientTransformation` for the RAdam optimizer."""

from __future__ import annotations

from torchopt.alias.utils import (
    _get_use_chain_flat,
    flip_sign_and_add_weight_decay,
    scale_by_neg_lr,
)
from torchopt.combine import chain
from torchopt.transform import scale_by_radam
from torchopt.typing import GradientTransformation, ScalarOrSchedule


__all__ = ['radam']


# pylint: disable-next=too-many-arguments
def radam(
    lr: ScalarOrSchedule = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-6,
    weight_decay: float = 0.0,
    *,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Create a functional version of the RAdam optimizer.

    RAdam is a variance of the adaptive learning rate rectified optimizer.

    References:
        - Liu, 2019: https://arxiv.org/abs/1908.03265

    Args:
        lr (float or callable, optional): This is a fixed global scaling factor or a learning rate
            scheduler. (default: :const:`1e-3`)
        betas (tuple of float, optional): Coefficients used for computing running averages of
            gradient and its square. (default: :const:`(0.9, 0.999)`)
        eps (float, optional): A small constant applied to the square root (as in the RAdam paper)
            to avoid dividing by zero when rescaling.
            (default: :const:`1e-6`)
        weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
            (default: :const:`0.0`)
        moment_requires_grad (bool, optional): If :data:`True` the momentums will be created with
            flag ``requires_grad=True``, this flag is often used in Meta-Learning algorithms.
            (default: :data:`False`)

    Returns:
        The corresponding :class:`GradientTransformation` instance.

    See Also:
        The functional optimizer wrapper :class:`torchopt.FuncOptimizer`.
    """
    b1, b2 = betas  # pylint: disable=invalid-name
    # pylint: disable=unneeded-not
    if not (callable(lr) or lr >= 0.0):  # pragma: no cover
        raise ValueError(f'Invalid learning rate: {lr}')
    if not 0 <= b1 <= 1:  # pragma: no cover
        raise ValueError(f'Invalid rho value: {b1}')
    if not 0 <= b2 <= 1:  # pragma: no cover
        raise ValueError(f'Invalid rho value: {b2}')
    if not eps >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not weight_decay >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    # pylint: enable=unneeded-not

    chain_fn = chain
    flip_sign_and_add_weight_decay_fn = flip_sign_and_add_weight_decay
    radam_scaler_fn = scale_by_radam
    scale_by_neg_lr_fn = scale_by_neg_lr

    if _get_use_chain_flat():  # default behavior
        chain_fn = chain_fn.flat  # type: ignore[attr-defined]
        flip_sign_and_add_weight_decay_fn = flip_sign_and_add_weight_decay_fn.flat  # type: ignore[attr-defined]
        radam_scaler_fn = radam_scaler_fn.flat  # type: ignore[attr-defined]
        scale_by_neg_lr_fn = scale_by_neg_lr_fn.flat  # type: ignore[attr-defined]

    return chain_fn(
        flip_sign_and_add_weight_decay_fn(weight_decay=weight_decay, maximize=False),
        radam_scaler_fn(
            b1=b1,
            b2=b2,
            eps=eps,
            moment_requires_grad=moment_requires_grad,
        ),
        scale_by_neg_lr_fn(lr),
    )
