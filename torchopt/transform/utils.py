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
"""Utilities for the preset transformations."""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Sequence

import torch

from torchopt import pytree
from torchopt.typing import TensorTree, Updates


__all__ = ['tree_map_flat', 'tree_map_flat_', 'inc_count', 'update_moment']


INT64_MAX = torch.iinfo(torch.int64).max


def tree_map_flat(
    func: Callable,
    flat_arg: Sequence[Any],
    *flat_args: Any,
    none_is_leaf: bool = False,
) -> Sequence[Any]:
    """Apply a function to each element of a flattened list."""
    if none_is_leaf:
        fn = func
    else:

        def fn(x: Any | None, *xs: Any) -> Any | None:
            return func(x, *xs) if x is not None else None

    return flat_arg.__class__(map(fn, flat_arg, *flat_args))  # type: ignore[call-arg]


def tree_map_flat_(
    func: Callable,
    flat_arg: Sequence[Any],
    *flat_args: Any,
    none_is_leaf: bool = False,
) -> Sequence[Any]:
    """Apply a function to each element of a flattened list and return the original list."""
    if none_is_leaf:
        fn = func
    else:

        def fn(x: Any | None, *xs: Any) -> Any | None:
            return func(x, *xs) if x is not None else None

    flat_results = map(fn, flat_arg, *flat_args)
    deque(flat_results, maxlen=0)  # consume and exhaust the iterable
    return flat_arg


def inc_count(updates: Updates, count: TensorTree) -> TensorTree:
    """Increment int counter by one.

    Returns:
        A counter incremented by one, or :data:`INT64_MAX` if the maximum precision is reached.
    """
    return _inc_count(
        updates=updates,
        count=count,
        already_flattened=False,
    )


def _inc_count_flat(updates: Updates, count: TensorTree) -> TensorTree:
    return _inc_count(
        updates=updates,
        count=count,
        already_flattened=True,
    )


def _inc_count(
    updates: Updates,
    count: TensorTree,
    *,
    already_flattened: bool = False,
) -> TensorTree:
    def f(c: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor:  # pylint: disable=invalid-name
        return c + (c != INT64_MAX).to(torch.int64) if g is not None else c

    if already_flattened:
        return tree_map_flat(f, count, updates, none_is_leaf=True)
    return pytree.tree_map(f, count, updates, none_is_leaf=True)


inc_count.flat = _inc_count_flat  # type: ignore[attr-defined]
inc_count.impl = _inc_count  # type: ignore[attr-defined]


def update_moment(
    updates: Updates,
    moments: TensorTree,
    decay: float,
    *,
    order: int,
    inplace: bool = True,
) -> TensorTree:
    """Compute the exponential moving average of the ``order``-th moment."""
    return _update_moment(
        updates,
        moments,
        decay,
        order=order,
        inplace=inplace,
        already_flattened=False,
    )


def _update_moment_flat(
    updates: Updates,
    moments: TensorTree,
    decay: float,
    *,
    order: int,
    inplace: bool = True,
) -> TensorTree:
    return _update_moment(
        updates,
        moments,
        decay,
        order=order,
        inplace=inplace,
        already_flattened=True,
    )


# pylint: disable-next=too-many-arguments
def _update_moment(
    updates: Updates,
    moments: TensorTree,
    decay: float,
    *,
    order: int,
    inplace: bool = True,
    already_flattened: bool = False,
) -> TensorTree:
    assert order in (1, 2)

    if inplace:
        if order == 2:
            if decay != 1.0:

                def f(g: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
                    return t.mul_(decay).addcmul_(g, g, value=1 - decay) if g is not None else t

            else:

                def f(g: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
                    return t.addcmul_(g, g) if g is not None else t

        else:
            if decay != 1.0:

                def f(g: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
                    return t.mul_(decay).add_(g, alpha=1 - decay) if g is not None else t

            else:

                def f(g: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
                    return t.add_(g) if g is not None else t

    else:
        if order == 2:
            if decay != 1.0:

                def f(g: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
                    return t.mul(decay).addcmul_(g, g, value=1 - decay) if g is not None else t

            else:

                def f(g: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
                    return t.addcmul(g, g) if g is not None else t

        else:
            if decay != 1.0:

                def f(g: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
                    return t.mul(decay).add_(g, alpha=1 - decay) if g is not None else t

            else:

                def f(g: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
                    return t.add(g) if g is not None else t

    if already_flattened:
        return tree_map_flat(f, updates, moments, none_is_leaf=True)
    return pytree.tree_map(f, updates, moments, none_is_leaf=True)


update_moment.flat = _update_moment_flat  # type: ignore[attr-defined]
update_moment.impl = _update_moment  # type: ignore[attr-defined]
