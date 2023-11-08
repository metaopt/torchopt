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

"""Preset transformations for scaling updates by Adamax."""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.base import GradientTransformation
from torchopt.transform.utils import tree_map_flat, update_moment
from torchopt.typing import OptState, Params, Updates


__all__ = ['scale_by_adamax']


class ScaleByAdamaxState(NamedTuple):
    """State for the Adamax algorithm."""

    mu: Updates
    nu: Updates
    t: int


def scale_by_adamax(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """A Adam algorithm variation.

    References:
        - Kingma et al., 2014: https://arxiv.org/abs/1412.6980

    Args:
        b1 (float, optional): Decay rate for the exponentially weighted average of grads.
            (default: :const:`0.9`)
        b2 (float, optional): Decay rate for the exponentially weighted average of squared grads.
            (default: :const:`0.999`)
        eps (float, optional): Term added to the denominator to improve numerical stability.
            (default: :const:`1e-6`)
        moment_requires_grad (bool, optional): If :data:`True`, states will be created with flag
            ``requires_grad = True``. (default: :data:`False`)

    Returns:
        An (init_fn, update_fn) tuple.
    """
    return _scale_by_adamax(
        b1=b1,
        b2=b2,
        eps=eps,
        moment_requires_grad=moment_requires_grad,
        already_flattened=False,
    )


def _scale_by_adamax_flat(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    return _scale_by_adamax(
        b1=b1,
        b2=b2,
        eps=eps,
        moment_requires_grad=moment_requires_grad,
        already_flattened=True,
    )


def _scale_by_adamax(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    moment_requires_grad: bool = False,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    # pylint: disable=unneeded-not
    if not eps >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= b1 < 1.0:  # pragma: no cover
        raise ValueError(f'Invalid b1 parameter at index 0: {b1}')
    if not 0.0 <= b2 < 1.0:  # pragma: no cover
        raise ValueError(f'Invalid b1 parameter at index 0: {b2}')
    # pylint: enable=unneeded-not

    if already_flattened:  # noqa: SIM108
        tree_map = tree_map_flat
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]

    def init_fn(params: Params) -> OptState:
        mu = tree_map(  # first moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad),
            params,
        )
        nu = tree_map(  # second moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad),
            params,
        )
        return ScaleByAdamaxState(mu=mu, nu=nu, t=1)

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,  # pylint: disable=unused-argument
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        mu = update_moment.impl(  # type: ignore[attr-defined]
            updates,
            state.mu,
            b1,
            order=1,
            inplace=inplace,
            already_flattened=already_flattened,
        )

        def update_nu(n: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
            return torch.max(n.mul(b2), g.abs().add_(eps)) if g is not None else g

        nu = tree_map(update_nu, state.nu, updates)

        one_minus_b1_pow_t = 1 - b1**state.t

        def f(m: torch.Tensor, n: torch.Tensor | None) -> torch.Tensor:
            return m.div(n).div_(one_minus_b1_pow_t) if n is not None else m

        updates = tree_map(f, mu, nu)

        return updates, ScaleByAdamaxState(mu=mu, nu=nu, t=state.t + 1)

    return GradientTransformation(init_fn, update_fn)


scale_by_adamax.flat = _scale_by_adamax_flat  # type: ignore[attr-defined]
scale_by_adamax.impl = _scale_by_adamax  # type: ignore[attr-defined]
