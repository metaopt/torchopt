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

"""Preset transformations for scaling updates by RAdam."""

# pylint: disable=invalid-name

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.base import GradientTransformation
from torchopt.transform.utils import tree_map_flat, update_moment
from torchopt.typing import OptState, Params, Updates


__all__ = ['scale_by_radam']


class ScaleByRAdamState(NamedTuple):
    """State for the RAdam algorithm."""

    mu: Updates
    nu: Updates
    t: int


def scale_by_radam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Rescale updates according to the RAdam algorithm.

    References:
        - Liu, 2019: https://arxiv.org/abs/1908.03265

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
    return _scale_by_radam(
        b1=b1,
        b2=b2,
        eps=eps,
        moment_requires_grad=moment_requires_grad,
        already_flattened=False,
    )


def _scale_by_radam_flat(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-6,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    return _scale_by_radam(
        b1=b1,
        b2=b2,
        eps=eps,
        moment_requires_grad=moment_requires_grad,
        already_flattened=True,
    )


def _scale_by_radam(
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
        return ScaleByRAdamState(mu=mu, nu=nu, t=1)

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

        nu = update_moment.impl(  # type: ignore[attr-defined]
            updates,
            state.nu,
            b2,
            order=2,
            inplace=inplace,
            already_flattened=already_flattened,
        )

        rho_inf = 2 / (1 - b2) - 1
        one_minus_b1_pow_t = 1 - b1**state.t
        one_minus_b2_pow_t = 1 - b2**state.t
        rho = rho_inf - 2 * state.t * b2**state.t / one_minus_b2_pow_t

        if rho > 5:
            numerator = math.sqrt(
                one_minus_b2_pow_t
                * (rho - 4)
                * (rho - 2)
                * rho_inf
                / ((rho_inf - 4) * (rho_inf - 2) * rho),
            )
            if inplace:

                def f(
                    m: torch.Tensor,
                    v: torch.Tensor,
                ) -> torch.Tensor:
                    return m.mul(numerator / one_minus_b1_pow_t).div_(v.sqrt().add_(eps))

            else:

                def f(
                    m: torch.Tensor,
                    v: torch.Tensor,
                ) -> torch.Tensor:
                    return m.mul(numerator / one_minus_b1_pow_t).div(v.sqrt().add(eps))

        else:
            if inplace:

                def f(
                    m: torch.Tensor,
                    v: torch.Tensor,  # pylint: disable=unused-argument
                ) -> torch.Tensor:
                    return m.div(one_minus_b1_pow_t)

            else:

                def f(
                    m: torch.Tensor,
                    v: torch.Tensor,  # pylint: disable=unused-argument
                ) -> torch.Tensor:
                    return m.div(one_minus_b1_pow_t)

        updates = tree_map(f, mu, nu)

        return updates, ScaleByRAdamState(mu=mu, nu=nu, t=state.t + 1)

    return GradientTransformation(init_fn, update_fn)


scale_by_radam.flat = _scale_by_radam_flat  # type: ignore[attr-defined]
scale_by_radam.impl = _scale_by_radam  # type: ignore[attr-defined]
