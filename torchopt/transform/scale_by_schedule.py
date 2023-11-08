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
# https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
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
"""Preset transformation for scaling updates by learning rate schedules."""

from __future__ import annotations

from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.base import GradientTransformation
from torchopt.transform.utils import inc_count, tree_map_flat, tree_map_flat_
from torchopt.typing import Numeric, OptState, Params, Schedule, SequenceOfTensors, Updates


__all__ = ['scale_by_schedule']


class ScaleByScheduleState(NamedTuple):
    """Maintain count for scale scheduling."""

    count: SequenceOfTensors  # type: ignore


def scale_by_schedule(step_size_fn: Schedule) -> GradientTransformation:
    """Scale updates using a custom schedule for the ``step_size``.

    Args:
        step_size_fn (callable): A function that takes an update count as input and proposes the
            ``step_size`` to multiply the updates by.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """
    return _scale_by_schedule(step_size_fn=step_size_fn, already_flattened=False)


def _scale_by_schedule_flat(step_size_fn: Schedule) -> GradientTransformation:
    return _scale_by_schedule(step_size_fn=step_size_fn, already_flattened=True)


def _scale_by_schedule(
    step_size_fn: Schedule,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    if already_flattened:
        tree_map = tree_map_flat
        tree_map_ = tree_map_flat_
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]
        tree_map_ = pytree.tree_map_  # type: ignore[assignment]

    def init_fn(params: Params) -> OptState:
        zero = tree_map(  # count init
            lambda t: torch.zeros(1, dtype=torch.int64, device=t.device).squeeze_(),
            params,
        )
        return ScaleByScheduleState(count=zero)

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,  # pylint: disable=unused-argument
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        if inplace:
            # pylint: disable-next=invalid-name
            def f(c: Numeric, g: torch.Tensor | None) -> torch.Tensor | None:
                if g is None:
                    return g
                step_size = step_size_fn(c)
                return g.mul_(step_size)

            tree_map_(f, state.count, updates)

        else:
            # pylint: disable-next=invalid-name
            def f(c: Numeric, g: torch.Tensor | None) -> torch.Tensor | None:
                if g is None:
                    return g
                step_size = step_size_fn(c)
                return g.mul(step_size)

            updates = tree_map(f, state.count, updates)

        return (
            updates,
            ScaleByScheduleState(
                count=inc_count.impl(  # type: ignore[attr-defined]
                    updates,
                    state.count,
                    already_flattened=already_flattened,
                ),
            ),
        )

    return GradientTransformation(init_fn, update_fn)


scale_by_schedule.flat = _scale_by_schedule_flat  # type: ignore[attr-defined]
scale_by_schedule.impl = _scale_by_schedule  # type: ignore[attr-defined]
