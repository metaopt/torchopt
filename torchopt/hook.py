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
"""Hook utilities."""

from typing import Callable, Optional

import torch

from torchopt import pytree
from torchopt.base import EmptyState, GradientTransformation


__all__ = ['zero_nan_hook', 'nan_to_num_hook', 'nan_to_num', 'register_hook']


def zero_nan_hook(g: torch.Tensor) -> torch.Tensor:
    """A zero ``nan`` hook to replace ``nan`` with zero."""
    return g.nan_to_num(nan=0.0)


def nan_to_num_hook(
    nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Returns a ``nan`` to num hook to replace ``nan`` with given number."""

    def hook(g: torch.Tensor) -> torch.Tensor:
        """A zero ``nan`` hook to replace ``nan`` with given number."""
        return g.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)

    return hook


def nan_to_num(
    nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None
) -> GradientTransformation:
    """A gradient transformation that replaces gradient values of ``nan`` with given number.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(params):  # pylint: disable=unused-argument
        return EmptyState()

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        if inplace:

            def f(g):
                return g.nan_to_num_(nan=nan, posinf=posinf, neginf=neginf)

        else:

            def f(g):
                return g.nan_to_num(nan=nan, posinf=posinf, neginf=neginf)

        new_updates = pytree.tree_map(f, updates)
        return new_updates, state

    return GradientTransformation(init_fn, update_fn)


def register_hook(hook) -> GradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

    This function passes through the *gradient updates* unchanged.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(params):  # pylint: disable=unused-argument
        return EmptyState()

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        def f(g):
            return g.register_hook(hook)

        pytree.tree_map(f, updates)
        return updates, state

    return GradientTransformation(init_fn, update_fn)
