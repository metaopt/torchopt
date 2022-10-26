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
"""Preset transformations for scaling updates by exponential root mean-squared (RMS)."""

from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.base import GradientTransformation
from torchopt.transform.utils import tree_map_flat, update_moment
from torchopt.typing import Updates


__all__ = ['scale_by_rms']


class ScaleByRmsState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""

    nu: Updates


def scale_by_rms(
    alpha: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.0
) -> GradientTransformation:
    """Rescale updates by the root of the exp. moving avg of the square.

    References:
        [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    Args:
        alpha: (default: :const:`0.9`)
            Decay rate for the exponentially weighted average of squared grads.
        eps: (default: :const:`1e-8`)
            Term added to the denominator to improve numerical stability.
        initial_scale: (default: :const:`0.0`)
            Initial value for second moment

    Returns:
        An (init_fn, update_fn) tuple.
    """
    return _scale_by_rms(
        alpha=alpha,
        eps=eps,
        initial_scale=initial_scale,
        already_flattened=False,
    )


def _scale_by_rms_flat(
    alpha: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.0
) -> GradientTransformation:
    return _scale_by_rms(
        alpha=alpha,
        eps=eps,
        initial_scale=initial_scale,
        already_flattened=True,
    )


def _scale_by_rms(
    alpha: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    # pylint: disable=unneeded-not
    if not 0.0 <= alpha:
        raise ValueError(f'Invalid alpha value: {alpha}')
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    # pylint: enable=unneeded-not

    if already_flattened:
        tree_map = tree_map_flat
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]

    def init_fn(params):
        nu = tree_map(lambda n: torch.full_like(n, initial_scale), params)  # second moment
        return ScaleByRmsState(nu=nu)

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        nu = update_moment.impl(  # type: ignore[attr-defined]
            updates, state.nu, alpha, order=2, inplace=inplace, already_flattened=already_flattened
        )

        if inplace:

            def f(g, n):  # pylint: disable=invalid-name
                return g.div_(n.sqrt().add_(eps))

        else:

            def f(g, n):  # pylint: disable=invalid-name
                return g.div(n.sqrt().add(eps))

        updates = tree_map(f, updates, nu)
        return updates, ScaleByRmsState(nu=nu)

    return GradientTransformation(init_fn, update_fn)


scale_by_rms.flat = _scale_by_rms_flat  # type: ignore[attr-defined]
scale_by_rms.impl = _scale_by_rms  # type: ignore[attr-defined]
