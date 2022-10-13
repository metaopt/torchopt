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
"""Preset :class:`GradientTransformation` for the SGD optimizer."""

from torchopt.alias.utils import flip_sign_and_add_weight_decay, scale_by_neg_lr
from torchopt.combine import chain_flat
from torchopt.transform import trace
from torchopt.typing import GradientTransformation, ScalarOrSchedule


__all__ = ['sgd']


def sgd(
    lr: ScalarOrSchedule,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    *,
    moment_requires_grad: bool = False,
    maximize: bool = False,
) -> GradientTransformation:
    """The functional version of the canonical Stochastic Gradient Descent optimizer.

    This implements stochastic gradient descent. It also includes support for momentum, and nesterov
    acceleration, as these are standard practice when using stochastic gradient descent to train
    deep neural networks.

    References:
        - Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

    Args:
        lr: This is a fixed global scaling factor.
        momentum: (default: :const:`0.0`)
            The decay rate used by the momentum term. The momentum is not used when it is set to
            :const:`0.0`.
        weight_decay: (default: :const:`0.0`)
            Weight decay, add L2 penalty to parameters.
        dampening: (default: :const:`0.0`)
            Dampening for momentum.
        nesterov: (default: :data:`False`)
            Whether to use Nesterov momentum.
        moment_requires_grad: (default: :data:`False`)
            If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
            flag is often used in Meta-Learning algorithms.
        maximize: (default: :data:`False`)
            Maximize the params based on the objective, instead of minimizing.

    Returns:
        The corresponding :class:`GradientTransformation` instance.

    See Also:
        The functional optimizer wrapper :class:`torchopt.FuncOptimizer`.
    """
    # pylint: disable=unneeded-not
    if not (callable(lr) or 0.0 <= lr):
        raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= momentum:
        raise ValueError(f'Invalid momentum value: {momentum}')
    if not 0.0 <= weight_decay:
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        raise ValueError('Nesterov momentum requires a momentum and zero dampening')
    # pylint: enable=unneeded-not

    return chain_flat(
        flip_sign_and_add_weight_decay(weight_decay=weight_decay, maximize=maximize),
        trace.flat(  # type: ignore[attr-defined]
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            moment_requires_grad=moment_requires_grad,
        ),
        scale_by_neg_lr(lr),
    )
