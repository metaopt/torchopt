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
# https://github.com/deepmind/optax/blob/master/optax/_src/wrappers.py
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
"""Preset transformations for adding weight decay to updates."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import torch

from torchopt import pytree
from torchopt.base import EmptyState, GradientTransformation, identity
from torchopt.transform.utils import tree_map_flat, tree_map_flat_
from torchopt.typing import OptState, Params, Updates


__all__ = ['masked', 'add_decayed_weights']


class MaskedState(NamedTuple):
    """Maintain inner transform state for masked transformations."""

    inner_state: Any


class MaskedNode(NamedTuple):
    """A node used to mask out unspecified parts of a tree.

    This node is ignored when mapping functions across the tree e.g. using :func:`pytree.tree_map`
    since it is a container without children. It can therefore be used to mask out parts of a tree.
    """


def masked(
    inner: GradientTransformation,
    mask: OptState | Callable[[Params], OptState] | None = None,
) -> GradientTransformation:
    """Mask updates so only some are transformed, the rest are passed through.

    For example, it is common to skip weight decay for BatchNorm scale and all bias parameters. In
    many networks, these are the only parameters with only one dimension. So, you may create a mask
    function to mask these out as follows::
        mask_fn = lambda p: pytree.tree_map(lambda x: x.ndim != 1, p)
        weight_decay = torchopt.masked(torchopt.add_decayed_weights(0.001), mask_fn)
    You may alternatively create the mask pytree upfront::
        mask = pytree.tree_map(lambda x: x.ndim != 1, params)
        weight_decay = torchopt.masked(torchopt.add_decayed_weights(0.001), mask)
    For the ``inner`` transform, state will only be stored for the parameters that have a mask value
    of :data:`True`.

    Args:
        inner (GradientTransformation): Inner transformation to mask.
        mask (tree of Tensor, callable, or None, optional): A tree with same structure as (or a
            prefix of) the params tree, or a function that returns such a tree given the
            params/updates. The leaves should be booleans, :data:`True` for leaves/subtrees you want
            to apply the transformation to, and :data:`False` for those you want to skip.
            (default: :data:`None`)

    Returns:
        A :class:`GradientTransformation` wrapping ``inner``.
    """
    return _masked(inner=inner, mask=mask, already_flattened=False)


def _masked_flat(
    inner: GradientTransformation,
    mask: OptState | Callable[[Params], OptState] | None = None,
) -> GradientTransformation:
    return _masked(inner, mask, already_flattened=True)


def _masked(
    inner: GradientTransformation,
    mask: OptState | Callable[[Params], OptState] | None = None,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    if already_flattened:  # noqa: SIM108
        tree_map = tree_map_flat
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]

    def tree_mask(params: Params, mask_tree: OptState) -> Params:
        return tree_map(lambda p, m: p if m else MaskedNode(), params, mask_tree)

    def init_fn(params: Params) -> OptState:
        mask_tree = mask(params) if callable(mask) else mask
        masked_params = tree_mask(params, mask_tree)
        return MaskedState(inner_state=inner.init(masked_params))

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        mask_tree = mask(updates) if callable(mask) else mask
        masked_updates = tree_mask(updates, mask_tree)
        masked_params = None if params is None else tree_mask(params, mask_tree)

        new_masked_updates, new_inner_state = inner.update(
            masked_updates,
            state.inner_state,
            params=masked_params,
            inplace=inplace,
        )

        new_updates = tree_map(
            lambda old_u, new_u, m: new_u if m else old_u,
            updates,
            new_masked_updates,
            mask_tree,
        )
        return new_updates, MaskedState(inner_state=new_inner_state)

    return GradientTransformation(init_fn, update_fn)


masked.flat = _masked_flat  # type: ignore[attr-defined]
masked.impl = _masked  # type: ignore[attr-defined]


AddDecayedWeightsState = EmptyState


def add_decayed_weights(
    weight_decay: float = 0.0,
    mask: OptState | Callable[[Params], OptState] | None = None,
) -> GradientTransformation:
    """Add parameter scaled by `weight_decay`.

    Args:
        weight_decay (float, optional): A scalar weight decay rate. (default: :const:`0.0`)
        mask (tree of Tensor, callable, or None, optional): A tree with same structure as (or a
            prefix of) the params tree, or a function that returns such a tree given the
            params/updates. The leaves should be booleans, :data:`True` for leaves/subtrees you want
            to apply the transformation to, and :data:`False` for those you want to skip.
            (default: :data:`None`)

    Returns:
        An (init_fn, update_fn) tuple.
    """
    return _add_decayed_weights(
        weight_decay=weight_decay,
        mask=mask,
        already_flattened=False,
    )


def _add_decayed_weights_flat(
    weight_decay: float = 0.0,
    mask: OptState | Callable[[Params], OptState] | None = None,
) -> GradientTransformation:
    return _add_decayed_weights(
        weight_decay=weight_decay,
        mask=mask,
        already_flattened=True,
    )


def _add_decayed_weights(
    weight_decay: float = 0.0,
    mask: OptState | Callable[[Params], OptState] | None = None,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    # pylint: disable-next=unneeded-not
    if not weight_decay >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')

    if weight_decay == 0.0 and mask is None:
        return identity()

    if already_flattened:
        tree_map = tree_map_flat
        tree_map_ = tree_map_flat_
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]
        tree_map_ = pytree.tree_map_  # type: ignore[assignment]

    def init_fn(params: Params) -> OptState:  # pylint: disable=unused-argument
        return AddDecayedWeightsState()

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        assert params is not None, (
            'Parameters are required for weight decay. '
            'Call `update(updates, state, params=params)` instead.'
        )

        if inplace:

            def f(p: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                if g is None:
                    return g
                if g.requires_grad:
                    return g.add_(p, alpha=weight_decay)
                return g.add_(p.data, alpha=weight_decay)

            tree_map_(f, params, updates)

        else:

            def f(p: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                return g.add(p, alpha=weight_decay) if g is not None else g

            updates = tree_map(f, params, updates)

        return updates, state

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return masked.impl(  # type: ignore[attr-defined]
            inner=GradientTransformation(init_fn, update_fn),
            mask=mask,
            already_flattened=already_flattened,
        )
    return GradientTransformation(init_fn, update_fn)


add_decayed_weights.flat = _add_decayed_weights_flat  # type: ignore[attr-defined]
add_decayed_weights.impl = _add_decayed_weights  # type: ignore[attr-defined]
