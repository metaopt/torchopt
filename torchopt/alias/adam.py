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
# This file is modified from:
# https://github.com/deepmind/optax/blob/master/optax/_src/alias.py
# ==============================================================================
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Preset :class:`GradientTransformation` for the Adam optimizer."""

from typing import Tuple

from torchopt.alias.utils import flip_sign_and_add_weight_decay, scale_by_neg_lr
from torchopt.combine import chain_flat
from torchopt.transform import scale_by_accelerated_adam, scale_by_adam
from torchopt.typing import GradientTransformation, ScalarOrSchedule


__all__ = ['adam']


# pylint: disable-next=too-many-arguments
def adam(
    lr: ScalarOrSchedule = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
    maximize: bool = False,
    use_accelerated_op: bool = False,
) -> GradientTransformation:
    """The functional Adam optimizer.

    Adam is an SGD variant with learning rate adaptation. The *learning rate* used for each weight
    is computed from estimates of first- and second-order moments of the gradients (using suitable
    exponential moving averages).

    References:
        - Kingma et al, 2014: https://arxiv.org/abs/1412.6980

    Args:
        lr: (default: :const:`1e-3`)
            This is a fixed global scaling factor.
        betas: (default: :const:`(0.9, 0.999)`)
            Coefficients used for computing running averages of gradient and its square.
        eps: (default: :const:`1e-8`)
            A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        weight_decay: (default: :const:`0.0`)
            Weight decay, add L2 penalty to parameters.
        eps_root: (default: :data:`0.0`)
            A small constant applied to denominator inside the square root (as in RMSProp), to avoid
            dividing by zero when rescaling. This is needed for example when computing
            (meta-)gradients through Adam.
        moment_requires_grad: (default: :data:`False`)
            If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
            flag is often used in Meta-Learning algorithms.
        maximize: (default: :data:`False`)
            Maximize the params based on the objective, instead of minimizing.
        use_accelerated_op: (default: :data:`False`)
            If :data:`True` use our implemented fused operator.

    Returns:
        The corresponding :class:`GradientTransformation` instance.

    See Also:
        The functional optimizer wrapper :class:`torchopt.FuncOptimizer`.
    """
    b1, b2 = betas  # pylint: disable=invalid-name
    # pylint: disable=unneeded-not
    if not (callable(lr) or 0.0 <= lr):
        raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= b1 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 0: {b1}')
    if not 0.0 <= b2 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 1: {b2}')
    if not 0.0 <= weight_decay:
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    # pylint: enable=unneeded-not

    if use_accelerated_op:
        adam_scaler = scale_by_accelerated_adam.flat  # type: ignore[attr-defined]
    else:
        adam_scaler = scale_by_adam.flat  # type: ignore[attr-defined]

    return chain_flat(
        flip_sign_and_add_weight_decay(weight_decay=weight_decay, maximize=maximize),
        adam_scaler(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            moment_requires_grad=moment_requires_grad,
        ),
        scale_by_neg_lr(lr),
    )
