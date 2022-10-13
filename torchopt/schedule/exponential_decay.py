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

from torchopt.typing import Numeric, Scalar, Schedule


__all__ = ['exponential_decay']


def exponential_decay(
    init_value: Scalar,
    decay_rate: Scalar,
    transition_begin: int = 0,
    transition_steps: Optional[int] = None,
    end_value: Optional[float] = None,
) -> Schedule:
    """Constructs a schedule with either continuous or discrete exponential decay.
    Args:
    value: value to be held constant throughout.
    Returns:
    schedule: A function that maps step counts to values.
    """
    if transition_steps is not None and transition_steps <= 0:
        logging.info(
            'An linear schedule was set with a non-positive `transition_steps`'
            ' value; this will result in a constant schedule with value '
            '`init_value`.'
        )
        return lambda count: init_value

    if decay_rate == 0:
        logging.info(
            'An linear schedule was set with a zero `decay_rate` value; '
            'this will result in a constant schedule with value `init_value`.'
        )
        return lambda count: init_value

    if transition_begin < 0:
        logging.info(
            'An linear schedule was set with a negative `transition_begin` '
            'value; this will result in `transition_begin` falling back to `0`.'
        )
        transition_begin = 0

    if end_value is not None:
        clip_fn = max if decay_rate < 1.0 else min

    def schedule(count: Numeric) -> Numeric:
        decreased_count = count - transition_begin
        decayed_value = (
            init_value / (1 + (decreased_count - 1) * decay_rate)
            if decreased_count > 0
            else init_value
        )
        if end_value is not None:
            decayed_value = clip_fn(decayed_value, end_value)
        return decayed_value

    return schedule
