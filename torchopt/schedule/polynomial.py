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
"""Polynomial learning rate schedules."""

import logging

import numpy as np
import torch

from torchopt.typing import Numeric, Scalar, Schedule


__all__ = ['polynomial_schedule', 'linear_schedule']


def polynomial_schedule(
    init_value: Scalar,
    end_value: Scalar,
    power: Scalar,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Construct a schedule with polynomial transition from init to end value.

    Args:
        init_value (float or Tensor): Initial value for the scalar to be annealed.
        end_value (float or Tensor): End value of the scalar to be annealed.
        power (float or Tensor): The power of the polynomial used to transition from ``init`` to
            ``end``.
        transition_steps (int): Number of steps over which annealing takes place, the scalar starts
            changing at ``transition_begin`` steps and completes the transition by
            ``transition_begin + transition_steps`` steps. If ``transition_steps <= 0``, then the
            entire annealing process is disabled and the value is held fixed at ``init_value``.
        transition_begin (int, optional): Must be *positive*. After how many steps to start
            annealing (before this many steps the scalar value is held fixed at ``init_value``).
            (default: :const:`0`)

    Returns:
        schedule:
            A function that maps step counts to values.
    """
    if transition_steps <= 0:  # pragma: no cover
        logging.info(
            'A polynomial schedule was set with a non-positive `transition_steps` value; this '
            'results in a constant schedule with value `init_value`.',
        )
        return lambda count: init_value

    if transition_begin < 0:  # pragma: no cover
        logging.info(
            'An exponential schedule was set with a negative `transition_begin` value; this will '
            'result in `transition_begin` falling back to `0`.',
        )
        transition_begin = 0

    def schedule(count: Numeric) -> Numeric:
        clip = torch.clamp if isinstance(count, torch.Tensor) else np.clip
        count = clip(count - transition_begin, 0, transition_steps)  # type: ignore[operator]
        frac = 1.0 - count / transition_steps
        return (init_value - end_value) * (frac**power) + end_value

    return schedule


# Alias polynomial schedule to linear schedule for convenience.
def linear_schedule(
    init_value: Scalar,
    end_value: Scalar,
    transition_steps: int,
    transition_begin: int = 0,
) -> Schedule:
    """Alias polynomial schedule to linear schedule for convenience."""
    return polynomial_schedule(
        init_value=init_value,
        end_value=end_value,
        power=1,
        transition_steps=transition_steps,
        transition_begin=transition_begin,
    )
