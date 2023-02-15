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
r"""Utilities for the aliases of preset :class:`GradientTransformation`\s for optimizers."""

from __future__ import annotations

import threading

from torchopt import pytree
from torchopt.base import EmptyState, GradientTransformation, identity
from torchopt.transform import scale, scale_by_schedule
from torchopt.transform.utils import tree_map_flat, tree_map_flat_
from torchopt.typing import OptState, Params, ScalarOrSchedule, Updates


__all__ = ['flip_sign_and_add_weight_decay', 'scale_by_neg_lr']


__USE_CHAIN_FLAT_LOCK = threading.Lock()
__USE_CHAIN_FLAT = True


def _set_use_chain_flat(use_chain_flat: bool) -> None:  # only used for testing purposes
    global __USE_CHAIN_FLAT  # pylint: disable=global-statement
    with __USE_CHAIN_FLAT_LOCK:
        __USE_CHAIN_FLAT = use_chain_flat


def _get_use_chain_flat() -> bool:  # only used for testing purposes
    with __USE_CHAIN_FLAT_LOCK:
        return __USE_CHAIN_FLAT


def flip_sign_and_add_weight_decay(
    weight_decay: float = 0.0, maximize=False
) -> GradientTransformation:
    """Flip the sign of the updates and adds weight decay."""
    return _flip_sign_and_add_weight_decay(
        weight_decay=weight_decay,
        maximize=maximize,
        already_flattened=False,
    )


def _flip_sign_and_add_weight_decay_flat(
    weight_decay: float = 0.0, maximize=False
) -> GradientTransformation:
    """Flip the sign of the updates and adds weight decay."""
    return _flip_sign_and_add_weight_decay(
        weight_decay=weight_decay,
        maximize=maximize,
        already_flattened=True,
    )


def _flip_sign_and_add_weight_decay(
    weight_decay: float = 0.0,
    maximize=False,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    """Flip the sign of the updates and adds weight decay."""
    # pylint: disable-next=unneeded-not
    if not 0.0 <= weight_decay:  # pragma: no cover
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')

    if not maximize and weight_decay == 0.0:
        return identity()

    if already_flattened:
        tree_map = tree_map_flat
        tree_map_ = tree_map_flat_
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]
        tree_map_ = pytree.tree_map_  # type: ignore[assignment]

    def init_fn(params: Params) -> OptState:  # pylint: disable=unused-argument
        return EmptyState()

    if not maximize:  # gradient descent

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

                def f(g, p):
                    if g.requires_grad:
                        return g.add_(p, alpha=weight_decay)
                    return g.add_(p.data, alpha=weight_decay)

                updates = tree_map_(f, updates, params)

            else:

                def f(g, p):
                    return g.add(p, alpha=weight_decay)

                updates = tree_map(f, updates, params)

            return updates, state

    else:  # gradient ascent
        if weight_decay == 0.0:

            def update_fn(
                updates: Updates,
                state: OptState,
                *,
                params: Params | None = None,  # pylint: disable=unused-argument
                inplace: bool = True,
            ) -> tuple[Updates, OptState]:
                if inplace:

                    def f(g):
                        return g.neg_()

                    updates = tree_map_(f, updates)

                else:

                    def f(g):
                        return g.neg()

                    updates = tree_map(f, updates)

                return updates, state

        else:

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

                    def f(g, p):
                        if g.requires_grad:
                            return g.neg_().add_(p, alpha=weight_decay)
                        return g.neg_().add_(p.data, alpha=weight_decay)

                    updates = tree_map_(f, updates, params)

                else:

                    def f(g, p):
                        return g.neg().add_(p, alpha=weight_decay)

                    updates = tree_map(f, updates, params)

                return updates, state

    return GradientTransformation(init_fn, update_fn)


flip_sign_and_add_weight_decay.flat = _flip_sign_and_add_weight_decay_flat  # type: ignore[attr-defined]
flip_sign_and_add_weight_decay.impl = _flip_sign_and_add_weight_decay  # type: ignore[attr-defined]


def scale_by_neg_lr(lr: ScalarOrSchedule) -> GradientTransformation:
    """Scale the updates by the negative learning rate."""
    return _scale_by_neg_lr(lr=lr, already_flattened=False)


def _scale_by_neg_lr_flat(lr: ScalarOrSchedule) -> GradientTransformation:
    return _scale_by_neg_lr(lr=lr, already_flattened=True)


def _scale_by_neg_lr(lr: ScalarOrSchedule, *, already_flattened=False) -> GradientTransformation:
    if not (callable(lr) or 0.0 <= lr):  # pragma: no cover
        raise ValueError(f'Invalid learning rate: {lr}')

    if callable(lr):

        def schedule_wrapper(count):
            return -lr(count)  # type: ignore[operator]

        return scale_by_schedule.impl(  # type: ignore[attr-defined]
            schedule_wrapper,
            already_flattened=already_flattened,
        )
    return scale.impl(-lr, already_flattened=already_flattened)  # type: ignore[attr-defined]


scale_by_neg_lr.flat = _scale_by_neg_lr_flat  # type: ignore[attr-defined]
scale_by_neg_lr.impl = _scale_by_neg_lr  # type: ignore[attr-defined]
