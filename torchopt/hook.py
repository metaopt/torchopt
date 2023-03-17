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
"""Hook utilities."""

from __future__ import annotations

from typing import Callable

import torch

from torchopt import pytree
from torchopt.base import EmptyState, GradientTransformation
from torchopt.typing import OptState, Params, Updates


__all__ = ['zero_nan_hook', 'nan_to_num_hook', 'register_hook']


def zero_nan_hook(g: torch.Tensor) -> torch.Tensor:
    """Replace ``nan`` with zero."""
    return g.nan_to_num(nan=0.0)


def nan_to_num_hook(
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a ``nan`` to num hook to replace ``nan`` / ``+inf`` / ``-inf`` with the given numbers."""

    def hook(g: torch.Tensor) -> torch.Tensor:
        """Replace ``nan`` / ``+inf`` / ``-inf`` with the given numbers."""
        return g.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)

    return hook


def register_hook(hook: Callable[[torch.Tensor], torch.Tensor | None]) -> GradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

    This function passes through the *gradient updates* unchanged.

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
        inplace: bool = True,  # pylint: disable=unused-argument
    ) -> tuple[Updates, OptState]:
        def f(g: torch.Tensor) -> torch.utils.hooks.RemovableHandle:
            return g.register_hook(hook)

        pytree.tree_map_(f, updates)
        return updates, state

    return GradientTransformation(init_fn, update_fn)
