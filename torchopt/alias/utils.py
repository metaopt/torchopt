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
r"""Utilities for the aliases of preset :class:`GradientTransformation`\s for optimizers."""

from torchopt.base import EmptyState, GradientTransformation, identity
from torchopt.transform import scale, scale_by_schedule
from torchopt.transform.utils import tree_map_flat
from torchopt.typing import ScalarOrSchedule


__all__ = ['flip_sign_and_add_weight_decay', 'scale_by_neg_lr']


def flip_sign_and_add_weight_decay(weight_decay: float = 0.0, maximize=False):
    """Flips the sign of the updates and adds weight decay."""
    if not 0.0 <= weight_decay:  # pylint: disable=unneeded-not
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')

    if not maximize and weight_decay == 0.0:
        return identity()

    def init_fn(params):  # pylint: disable=unused-argument
        return EmptyState()

    if not maximize:  # gradient descent

        def update_fn(updates, state, *, params=None, inplace=True):
            assert params is not None, (
                'Parameters are required for weight decay. '
                'Call `update(updates, state, params=params)` instead.'
            )

            if inplace:

                def f(g, p):
                    if g.requires_grad:
                        return g.add_(p, alpha=weight_decay)
                    return g.add_(p.data, alpha=weight_decay)

            else:

                def f(g, p):
                    return g.add(p, alpha=weight_decay)

            updates = tree_map_flat(f, updates, params)
            return updates, state

    else:  # gradient ascent

        if weight_decay == 0.0:
            # pylint: disable-next=unused-argument
            def update_fn(updates, state, *, params=None, inplace=True):
                if inplace:

                    def f(g):
                        return g.neg_()

                else:

                    def f(g):
                        return g.neg()

                updates = tree_map_flat(f, updates)
                return updates, state

        else:

            def update_fn(updates, state, *, params=None, inplace=True):
                assert params is not None, (
                    'Parameters are required for weight decay. '
                    'Call `update(updates, state, params=params)` instead.'
                )

                if inplace:

                    def f(g, p):
                        if g is not None:
                            if g.requires_grad:
                                return g.neg_().add_(p, alpha=weight_decay)
                            return g.neg_().add_(p.data, alpha=weight_decay)
                        return None

                else:

                    def f(g, p):
                        return g.neg().add_(p, alpha=weight_decay)

                updates = tree_map_flat(f, updates, params)
                return updates, state

    return GradientTransformation(init_fn, update_fn)


def scale_by_neg_lr(lr: ScalarOrSchedule):
    """Scales the updates by the negative learning rate."""
    if not (callable(lr) or 0.0 <= lr):
        raise ValueError(f'Invalid learning rate: {lr}')

    if callable(lr):

        def schedule_wrapper(count):
            return -lr(count)  # type: ignore[operator]

        return scale_by_schedule.flat(schedule_wrapper)  # type: ignore[attr-defined]
    return scale.flat(-lr)  # type: ignore[attr-defined]
