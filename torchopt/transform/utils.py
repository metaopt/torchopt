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

from collections import deque
from typing import Any, Callable, Iterable, List

import torch

from torchopt import pytree
from torchopt.typing import TensorTree, Updates


__all__ = ['tree_map_flat', 'tree_map_flat_', 'inc_count', 'update_moment']


INT64_MAX = torch.iinfo(torch.int64).max


def tree_map_flat(func: Callable, *flat_args: Any, none_is_leaf: bool = False) -> List[Any]:
    """Apply a function to each element of a flattened list."""
    if none_is_leaf:
        fn = func
    else:

        def fn(x, *xs):
            return func(x, *xs) if x is not None else None

    return list(map(fn, *flat_args))


def tree_map_flat_(
    func: Callable, flat_arg: Iterable[Any], *flat_args: Any, none_is_leaf: bool = False
) -> Iterable[Any]:
    """Apply a function to each element of a flattened list."""
    if none_is_leaf:
        fn = func
    else:

        def fn(x, *xs):
            return func(x, *xs) if x is not None else None

    flat_results = map(fn, flat_arg, *flat_args)
    deque(flat_results, maxlen=0)  # consume and exhaust the iterable
    return flat_arg


def inc_count(updates: Updates, count: TensorTree) -> TensorTree:
    """Increments int counter by one.

    Returns:
        A counter incremented by one, or :data:`INT64_MAX` if the maximum precision is reached.
    """
    return _inc_count(updates=updates, count=count, already_flattened=False)


def _inc_count_flat(updates: Updates, count: TensorTree) -> TensorTree:
    return _inc_count(updates=updates, count=count, already_flattened=True)


def _inc_count(
    updates: Updates, count: TensorTree, *, already_flattened: bool = False
) -> TensorTree:
    def f(c, g):  # pylint: disable=invalid-name
        return c + (c != INT64_MAX).to(torch.int64) if g is not None else c

    if already_flattened:
        return tree_map_flat(f, count, updates)
    return pytree.tree_map(f, count, updates)


inc_count.flat = _inc_count_flat  # type: ignore[attr-defined]
inc_count.impl = _inc_count  # type: ignore[attr-defined]


def update_moment(updates, moments, decay, *, order, inplace=True):
    """Compute the exponential moving average of the ``order``-th moment."""
    return _update_moment(
        updates, moments, decay, order=order, inplace=inplace, already_flattened=False
    )


def _update_moment_flat(updates, moments, decay, *order, inplace=True):
    return _update_moment(
        updates, moments, decay, order=order, inplace=inplace, already_flattened=True
    )


def _update_moment(updates, moments, decay, *, order, inplace=True, already_flattened=False):
    assert order in (1, 2)

    if inplace:

        if order == 2:

            def f(g, t):
                return t.mul_(decay).addcmul_(g, g, value=1 - decay) if g is not None else t

        else:

            def f(g, t):
                return t.mul_(decay).add_(g, alpha=1 - decay) if g is not None else t

    else:

        if order == 2:

            def f(g, t):
                return t.mul(decay).addcmul_(g, g, value=1 - decay) if g is not None else t

        else:

            def f(g, t):
                return t.mul(decay).add_(g, alpha=1 - decay) if g is not None else t

    if already_flattened:
        return tree_map_flat(f, updates, moments)
    return pytree.tree_map(f, updates, moments, none_is_leaf=True)


update_moment.flat = _update_moment_flat  # type: ignore[attr-defined]
update_moment.impl = _update_moment  # type: ignore[attr-defined]
