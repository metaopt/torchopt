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
"""Preset transformation for scaling updates by learning rate."""

from __future__ import annotations

import torch

from torchopt import pytree
from torchopt.base import EmptyState, GradientTransformation
from torchopt.transform.utils import tree_map_flat, tree_map_flat_
from torchopt.typing import OptState, Params, Updates


__all__ = ['scale']


ScaleState = EmptyState


def scale(step_size: float) -> GradientTransformation:
    """Scale updates by some fixed scalar ``step_size``.

    Args:
        step_size (float): A scalar corresponding to a fixed scaling factor for updates.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """
    return _scale(step_size=step_size, already_flattened=False)


def _scale_flat(step_size: float) -> GradientTransformation:
    return _scale(step_size=step_size, already_flattened=True)


def _scale(
    step_size: float,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    if already_flattened:
        tree_map = tree_map_flat
        tree_map_ = tree_map_flat_
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]
        tree_map_ = pytree.tree_map_  # type: ignore[assignment]

    def init_fn(params: Params) -> OptState:  # pylint: disable=unused-argument
        return ScaleState()

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,  # pylint: disable=unused-argument
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        if inplace:

            def f(g: torch.Tensor) -> torch.Tensor:
                return g.mul_(step_size)

            updates = tree_map_(f, updates)

        else:

            def f(g: torch.Tensor) -> torch.Tensor:
                return g.mul(step_size)

            updates = tree_map(f, updates)

        return updates, state

    return GradientTransformation(init_fn, update_fn)


scale.flat = _scale_flat  # type: ignore[attr-defined]
scale.impl = _scale  # type: ignore[attr-defined]
