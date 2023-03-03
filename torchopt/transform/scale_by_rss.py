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
"""Preset transformations for scaling updates by the root of the sum of all squared gradients."""

from __future__ import annotations

from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.base import GradientTransformation
from torchopt.transform.utils import tree_map_flat
from torchopt.typing import OptState, Params, Updates


__all__ = ['scale_by_rss']


class ScaleByRssState(NamedTuple):
    """State holding the sum of gradient squares to date."""

    sum_of_squares: Updates


def scale_by_rss(
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7,
) -> GradientTransformation:
    """Rescale updates by the root of the sum of all squared gradients to date.

    References:
        [Duchi et al, 2011](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
        [McMahan et al., 2010](https://arxiv.org/abs/1002.4908)

    Args:
        initial_accumulator_value: Starting value for accumulators, must be >= 0.
        eps: A small floating point value to avoid zero denominator.

    Returns:
        An (init_fn, update_fn) tuple.
    """
    return _scale_by_rss(
        initial_accumulator_value=initial_accumulator_value,
        eps=eps,
        already_flattened=False,
    )


def _scale_by_rss_flat(
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7,
) -> GradientTransformation:
    return _scale_by_rss(
        initial_accumulator_value=initial_accumulator_value,
        eps=eps,
        already_flattened=True,
    )


def _scale_by_rss(
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    tree_map = tree_map_flat if already_flattened else pytree.tree_map

    def init_fn(params: Params) -> OptState:
        sum_of_squares = tree_map(lambda t: torch.full_like(t, initial_accumulator_value), params)
        return ScaleByRssState(sum_of_squares=sum_of_squares)

    def update_fn(
        updates: Updates,
        state: OptState,
        params: Params | None = None,
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:  # pylint: disable=unused-argument
        del params
        sum_of_squares = tree_map(
            lambda g, t: (g.conj() * g).real + t, updates, state.sum_of_squares
        )

        if inplace:

            def f(t: torch.Tensor) -> torch.Tensor:
                return t.add_(eps).rsqrt_() if t > 0.0 else 0.0

        else:

            def f(t: torch.Tensor) -> torch.Tensor:
                return t.add(eps).rsqrt() if t > 0.0 else 0.0

        inv_sqrt_g_square = tree_map(f, sum_of_squares)
        updates = tree_map(lambda scale, g: scale * g, inv_sqrt_g_square, updates)
        return updates, ScaleByRssState(sum_of_squares=sum_of_squares)

    return GradientTransformation(init_fn, update_fn)


scale_by_rss.flat = _scale_by_rss_flat  # type: ignore[attr-defined]
scale_by_rss.impl = _scale_by_rss  # type: ignore[attr-defined]