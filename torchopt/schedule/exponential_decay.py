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
import math
from typing import Optional

from torchopt.typing import Numeric, Scalar, Schedule


__all__ = ['exponential_decay']


# pylint: disable-next=too-many-arguments
def exponential_decay(
    init_value: Scalar,
    decay_rate: Scalar,
    transition_begin: int = 0,
    transition_steps: int = 1,
    staircase: bool = False,
    end_value: Optional[float] = None,
) -> Schedule:
    """Construct a schedule with either continuous or discrete exponential decay.

    This function applies an exponential decay function to a provided initial value. The function
    returns the decayed value as follows:

    .. code-block:: python

        decayed_value = init_value * decay_rate**(count / transition_steps)

    If the argument ``staircase`` is :data:`True`, then ``count / transition_steps`` is an integer
    division and the decayed value follows a staircase function.

    Args:
        init_value (float or Tensor): Initial value for the scalar to be annealed.
        decay_rate (float or Tensor): The decay rate.
        transition_begin (int, optional): Must be *positive*. After how many steps to start
            annealing (before this many steps the scalar value is held fixed at ``init_value``).
            (default: :const:`0`)
        transition_steps (int, optional): Number of steps over which annealing takes place, the
            scalar starts changing at ``transition_begin`` steps and completes the transition by
            ``transition_begin + transition_steps`` steps. If ``transition_steps <= 0``, then the
            entire annealing process is disabled and the value is held fixed at ``init_value``.
            (default: :const:`1`)
        staircase (bool, optional): If :data:`True`, decay the scalar at discrete intervals.
            (default: :data:`False`)
        end_value (float or Tensor, optional): End value of the scalar to be annealed.
            (default: :data:`None`)

    Returns:
        schedule: A function that maps step counts to values.
    """
    if transition_steps is not None and transition_steps <= 0:  # pragma: no cover
        logging.info(
            'An exponential schedule was set with a non-positive `transition_steps`'
            ' value; this will result in a constant schedule with value '
            '`init_value`.',
        )
        return lambda count: init_value

    if decay_rate == 0:  # pragma: no cover
        logging.info(
            'An exponential schedule was set with a zero `decay_rate` value; '
            'this will result in a constant schedule with value `init_value`.',
        )
        return lambda count: init_value

    if transition_begin < 0:  # pragma: no cover
        logging.info(
            'An exponential schedule was set with a negative `transition_begin` '
            'value; this will result in `transition_begin` falling back to `0`.',
        )
        transition_begin = 0

    if end_value is not None:  # pragma: no cover
        clip_fn = max if decay_rate < 1.0 else min

    def schedule(count: Numeric) -> Numeric:
        decreased_count = count - transition_begin
        p = decreased_count / transition_steps
        if staircase:
            p = math.floor(p)
        decayed_value = init_value if decreased_count <= 0.0 else init_value * (decay_rate**p)
        if end_value is not None:
            return clip_fn(decayed_value, end_value)
        return decayed_value

    return schedule
