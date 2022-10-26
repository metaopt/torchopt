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
# https://github.com/deepmind/optax/blob/master/optax/_src/alias.py
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
"""Utilities to define a chained transformation."""

from torchopt import pytree
from torchopt.base import ChainedGradientTransformation, GradientTransformation, identity
from torchopt.typing import Updates


__all__ = ['chain', 'chain_flat']


def chain(*transformations: GradientTransformation) -> GradientTransformation:
    """Applies a list of chainable update transformations.

    Given a sequence of chainable transforms, :func:`chain` returns an :func:`init_fn` that
    constructs a ``state`` by concatenating the states of the individual transforms, and returns an
    :func:`update_fn` which chains the update transformations feeding the appropriate state to each.

    Args:
        *transformations:
            A sequence of chainable ``(init_fn, update_fn)`` tuples.

    Returns:
        A single ``(init_fn, update_fn)`` tuple.
    """
    if len(transformations) == 0:
        return identity()
    if len(transformations) == 1:
        return transformations[0]
    return ChainedGradientTransformation(*transformations)


def chain_flat(*transformations: GradientTransformation) -> GradientTransformation:
    """Wraps around the inner transformations that manipulates the flattened tree structure (:class:``list``).

    Args:
        *transformations:
            A sequence of chainable ``(init_fn, update_fn)`` tuples.

    Returns:
        A single ``(init_fn, update_fn)`` tuple.
    """
    if len(transformations) == 0:
        return identity()
    if len(transformations) == 1:
        inner = transformations[0]
    else:
        inner = chain(*transformations)

    def init_fn(params):
        return inner.init(pytree.tree_leaves(params, none_is_leaf=True))

    def update_fn(updates, state, *, params=None, inplace=True):
        flat_updates, treespec = pytree.tree_flatten(updates, none_is_leaf=True)
        if params is not None:
            flat_params = pytree.tree_leaves(params, none_is_leaf=True)
        else:
            flat_params = None

        flat_updates, state = inner.update(flat_updates, state, params=flat_params, inplace=inplace)
        updates: Updates
        updates = pytree.tree_unflatten(treespec, flat_updates)
        return updates, state

    return GradientTransformation(init_fn, update_fn)


chain.flat = chain_flat  # type: ignore[attr-defined]
