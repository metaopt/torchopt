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
"""Preset :class:`GradientTransformation` for the RMSProp optimizer."""

from torchopt.alias.utils import flip_sign_and_add_weight_decay, scale_by_neg_lr
from torchopt.combine import chain_flat
from torchopt.transform import scale_by_rms, scale_by_stddev, trace
from torchopt.typing import GradientTransformation, ScalarOrSchedule


__all__ = ['rmsprop']


# pylint: disable-next=too-many-arguments
def rmsprop(
    lr: ScalarOrSchedule = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    centered: bool = False,
    *,
    initial_scale: float = 0.0,
    nesterov: bool = False,
    maximize: bool = False,
) -> GradientTransformation:
    """The functional version of the RMSProp optimizer.

    RMSProp is an SGD variant with learning rate adaptation. The *learning rate* used for each
    weight is scaled by a suitable estimate of the magnitude of the gradients on previous steps.
    Several variants of RMSProp can be found in the literature. This alias provides an easy to
    configure RMSProp optimizer that can be used to switch between several of these variants.

    References:
        - Tieleman and Hinton, 2012: http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
        - Graves, 2013: https://arxiv.org/abs/1308.0850

    Args:
        lr: (default: :const:`1e-2`)
            This is a fixed global scaling factor.
        alpha: (default: :const:`0.99`)
            Smoothing constant, the decay used to track the magnitude of previous gradients.
        eps: (default: :const:`1e-8`)
            A small numerical constant to avoid dividing by zero when rescaling.
        weight_decay: (default: :const:`0.0`)
            Weight decay, add L2 penalty to parameters.
        momentum: (default: :const:`0.0`)
            The decay rate used by the momentum term. The momentum is not used when it is set to
            :const:`0.0`.
        centered: (default: :data:`False`)
            If :data:`True`, use the variance of the past gradients to rescale the latest
            gradients.
        initial_scale: (default: :data:`0.0`)
            Initialization of accumulators tracking the magnitude of previous updates. PyTorch
            uses :data:`0.0`, TensorFlow 1.x uses :data:`1.0`. When reproducing results from a
            paper, verify the value used by the authors.
        nesterov: (default: :data:`False`)
            Whether to use Nesterov momentum.
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
    if not 0.0 <= alpha:
        raise ValueError(f'Invalid alpha value: {alpha}')
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= momentum:
        raise ValueError(f'Invalid momentum value: {momentum}')
    if not 0.0 <= weight_decay:
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    # pylint: enable=unneeded-not

    if centered:
        rmsprop_scaler = scale_by_stddev.flat  # type: ignore[attr-defined]
    else:
        rmsprop_scaler = scale_by_rms.flat  # type: ignore[attr-defined]

    return chain_flat(
        flip_sign_and_add_weight_decay(weight_decay=weight_decay, maximize=maximize),
        rmsprop_scaler(
            alpha=alpha,
            eps=eps,
            initial_scale=initial_scale,
        ),
        trace.flat(momentum=momentum, nesterov=nesterov),  # type: ignore[attr-defined]
        scale_by_neg_lr(lr),
    )
