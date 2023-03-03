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
"""Preset :class:`GradientTransformation` for the AdaGrad optimizer."""

from torchopt.alias.utils import flip_sign_and_add_weight_decay, scale_by_neg_lr
from torchopt.combine import chain_flat
from torchopt.transform import scale_by_rss
from torchopt.typing import GradientTransformation, Scalar


__all__ = ['adagrad']


# pylint: disable-next=too-many-arguments
def adagrad(
    lr: Scalar = 1e-2,
    lr_decay: float = 0.0,
    weight_decay: float = 0.0,
    initial_accumulator_value: float = 0.0,
    eps: float = 1e-10,
    *,
    maximize: bool = False,
) -> GradientTransformation:
    """The functional AdaGrad optimizer.

    AdaGrad is an algorithm for gradient based optimization that anneals the learning rate for each
    parameter during the course of training.
    WARNING: AdaGrad's main limit is the monotonic accumulation of squared gradients in the
    denominator: since all terms are >0, the sum keeps growing during training and the learning rate
    eventually becomes very small.

    References:
        Duchi et al, 2011: https://jmlr.org/papers/v12/duchi11a.html

    Args:
        lr: (default: :const:`1e-3`)
            This is a fixed global scaling factor.
        lr_decay: (default: :const:`0.0`)
            Learning rate decay.
        weight_decay: (default: :const:`0.0`)
            Weight decay, add L2 penalty to parameters.
        initial_accumulator_value: (default: :const:`0.0`)
            Initial value for the accumulator.
        eps: (default: :const:`1e-10`)
            A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        maximize: (default: :data:`False`)
            Maximize the params based on the objective, instead of minimizing.
        use_accelerated_op: (default: :data:`False`)
            If :data:`True` use our implemented fused operator.

    Returns:
        The corresponding :class:`GradientTransformation` instance.

    See Also:
        The functional optimizer wrapper :class:`torchopt.FuncOptimizer`.
    """
    # pylint: disable=unneeded-not
    if not (callable(lr) or lr >= 0.0):
        raise ValueError(f'Invalid learning rate: {lr}')
    if not eps >= 0.0:
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not lr_decay >= 0.0:
        raise ValueError(f'Invalid lr_decay value: {lr_decay}')
    if not weight_decay >= 0.0:
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    # pylint: enable=unneeded-not

    return chain_flat(
        flip_sign_and_add_weight_decay(weight_decay=weight_decay, maximize=maximize),
        scale_by_rss.flat(initial_accumulator_value=initial_accumulator_value, eps=eps),  # type: ignore[attr-defined]
        # scale_by_neg_lr(exponential_decay(init_value=lr, decay_rate=lr_decay)),
        scale_by_neg_lr(lr),
    )
