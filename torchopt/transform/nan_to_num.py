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
"""Preset transformations that replaces updates with non-finite values to the given numbers."""

from __future__ import annotations

import torch

from torchopt import pytree
from torchopt.base import EmptyState, GradientTransformation
from torchopt.typing import OptState, Params, Updates


def nan_to_num(
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> GradientTransformation:
    """Replace updates with values ``nan`` / ``+inf`` / ``-inf`` to the given numbers.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(params: Params) -> OptState:  # pylint: disable=unused-argument
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,  # pylint: disable=unused-argument
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        if inplace:

            def f(g: torch.Tensor) -> torch.Tensor:
                return g.nan_to_num_(nan=nan, posinf=posinf, neginf=neginf)

        else:

            def f(g: torch.Tensor) -> torch.Tensor:
                return g.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)

        new_updates = pytree.tree_map(f, updates)
        return new_updates, state

    return GradientTransformation(init_fn, update_fn)
