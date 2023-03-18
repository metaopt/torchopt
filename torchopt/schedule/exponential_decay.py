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
# https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py
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
"""Exponential learning rate decay."""

import logging
from typing import Optional

import torch

from torchopt.typing import Numeric, Scalar, Schedule


__all__ = ['exponential_decay']


# pylint: disable-next=too-many-arguments
def exponential_decay(
    init_value: Scalar,
    decay_rate: Scalar,
    transition_begin: int = 0,
    transition_steps: Optional[int] = None,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> Schedule:
    """Constructs a schedule with either continuous or discrete exponential decay.

    This function applies an exponential decay function to a provided initial
    value. The function returns the decayed value as follows:
    ```
    decayed_value = init_value * decay_rate ^ (count / transition_steps)
    ```
    If the argument `staircase` is `True`, then `count / transition_steps` is
    an integer division and the decayed value follows a staircase function.
    Args:
        init_value: the initial learning rate.
        decay_rate: must not be zero. The decay rate.
        transition_begin: must be positive. After how many steps to start annealing
            (before this many steps the scalar value is held fixed at `init_value`).
        transition_steps: must be positive. See the decay computation above.
        staircase: if `True`, decay the values at discrete intervals.
        end_value: the value at which the exponential decay stops. When
            `decay_rate` < 1, `end_value` is treated as a lower bound, otherwise as
            an upper bound. Has no effect when `decay_rate` = 0.

    Returns:
        schedule: A function that maps step counts to values.
    """
    if transition_steps is not None and transition_steps <= 0:
        logging.info(
            'An exponential schedule was set with a non-positive `transition_steps`'
            ' value; this will result in a constant schedule with value '
            '`init_value`.',
        )
        return lambda count: init_value

    if decay_rate == 0:
        logging.info(
            'An exponential schedule was set with a zero `decay_rate` value; '
            'this will result in a constant schedule with value `init_value`.',
        )
        return lambda count: init_value

    if transition_begin < 0:
        logging.info(
            'An exponential schedule was set with a negative `transition_begin` '
            'value; this will result in `transition_begin` falling back to `0`.',
        )
        transition_begin = 0

    if end_value is not None:
        clip_fn = torch.maximum if decay_rate < 1.0 else torch.minimum

    def schedule(count: Numeric) -> Numeric:
        decreased_count = count - transition_begin
        if transition_steps is not None:
            p = decreased_count / transition_steps

            if staircase:
                p = torch.floor(torch.tensor(p))

            decayed_value = torch.where(
                torch.tensor(decreased_count) <= 0,
                torch.tensor(init_value),
                torch.tensor(init_value) * torch.pow(torch.tensor(decay_rate), p),
            )
        else:
            decayed_value = torch.tensor(init_value) * torch.pow(
                torch.tensor(decay_rate),
                decreased_count,
            )
        if end_value is not None:
            return clip_fn(decayed_value, end_value)
        return decayed_value

    return schedule
