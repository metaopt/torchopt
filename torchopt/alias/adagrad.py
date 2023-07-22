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

import logging

from torchopt.alias.utils import (
    _get_use_chain_flat,
    flip_sign_and_add_weight_decay,
    scale_by_neg_lr,
)
from torchopt.combine import chain
from torchopt.transform import scale_by_rss, scale_by_schedule
from torchopt.typing import GradientTransformation, Numeric, Scalar, ScalarOrSchedule, Schedule


__all__ = ['adagrad']


def _adagrad_lr_schedule(
    decay_rate: Scalar,
    transition_begin: int = 0,
) -> Schedule:
    """Construct a schedule dedicated to AdaGrad optimizer.

    This function applies an learning rate decay function to a provided initial value. The function
    returns the decayed value as follows:

    .. code-block:: python

        decayed_value = init_value / (1 + count * decay_rate)

    Args:
        decay_rate (float): The decay rate.
        transition_begin (int, optional): Must be *positive*. After how many steps to start
            annealing. (default: :const:`0`)

    Returns:
        schedule: A function that maps step counts to values.
    """
    if transition_begin < 0:  # pragma: no cover
        logging.info(
            'The AdaGrad learning rate schedule was set with a negative `transition_begin` '
            'value; this will result in `transition_begin` falling back to `0`.',
        )
        transition_begin = 0

    def schedule(count: Numeric) -> Numeric:
        decreased_count = count - transition_begin
        return 1 / (1 + decay_rate * decreased_count)

    return schedule


# pylint: disable-next=too-many-arguments
def adagrad(
    lr: ScalarOrSchedule = 1e-2,
    lr_decay: float = 0.0,
    weight_decay: float = 0.0,
    initial_accumulator_value: float = 0.0,
    eps: float = 1e-10,
    *,
    maximize: bool = False,
) -> GradientTransformation:
    """Create a functional version of the AdaGrad optimizer.

    AdaGrad is an algorithm for gradient based optimization that anneals the learning rate for each
    parameter during the course of training.

    .. warning::
        AdaGrad's main limit is the monotonic accumulation of squared gradients in the denominator.
        Since all terms are ``> 0``, the sum keeps growing during training, and the learning rate
        eventually becomes very small.

    References:
        Duchi et al., 2011: https://jmlr.org/papers/v12/duchi11a.html

    Args:
        lr (float or callable, optional): This is a fixed global scaling factor or a learning rate
            scheduler. (default: :const:`1e-2`)
        lr_decay (float, optional): Learning rate decay. (default: :const:`0.0`)
        weight_decay (float, optional): Weight decay, add L2 penalty to parameters.
            (default: :const:`0.0`)
        initial_accumulator_value (float, optional): Initial value for the accumulator.
            (default: :const:`0.0`)
        eps (float, optional): A small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
            (default: :const:`1e-10`)
        maximize (bool, optional): Maximize the params based on the objective, instead of minimizing.
            (default: :data:`False`)

    Returns:
        The corresponding :class:`GradientTransformation` instance.

    See Also:
        The functional optimizer wrapper :class:`torchopt.FuncOptimizer`.
    """
    # pylint: disable=unneeded-not
    if not (callable(lr) or lr >= 0.0):  # pragma: no cover
        raise ValueError(f'Invalid learning rate: {lr}')
    if not lr_decay >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid lr_decay value: {lr_decay}')
    if not weight_decay >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    if not initial_accumulator_value >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid initial_accumulator_value value: {initial_accumulator_value}')
    if not eps >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid epsilon value: {eps}')
    # pylint: enable=unneeded-not

    chain_fn = chain
    flip_sign_and_add_weight_decay_fn = flip_sign_and_add_weight_decay
    adagrad_scaler_fn = scale_by_rss
    scale_by_neg_lr_fn = scale_by_neg_lr
    scale_by_schedule_fn = scale_by_schedule

    if _get_use_chain_flat():  # default behavior
        chain_fn = chain_fn.flat  # type: ignore[attr-defined]
        flip_sign_and_add_weight_decay_fn = flip_sign_and_add_weight_decay_fn.flat  # type: ignore[attr-defined]
        adagrad_scaler_fn = adagrad_scaler_fn.flat  # type: ignore[attr-defined]
        scale_by_neg_lr_fn = scale_by_neg_lr_fn.flat  # type: ignore[attr-defined]
        scale_by_schedule_fn = scale_by_schedule_fn.flat  # type: ignore[attr-defined]

    return chain_fn(
        flip_sign_and_add_weight_decay_fn(weight_decay=weight_decay, maximize=maximize),
        adagrad_scaler_fn(
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
        ),
        scale_by_schedule_fn(
            step_size_fn=_adagrad_lr_schedule(
                decay_rate=lr_decay,
                transition_begin=0,
            ),
        ),
        scale_by_neg_lr_fn(lr),
    )
