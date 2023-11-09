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

"""Preset transformations for scaling updates by Adam."""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.base import GradientTransformation
from torchopt.transform.utils import tree_map_flat, update_moment
from torchopt.typing import OptState, Params, Updates


__all__ = ['scale_by_adadelta']


class ScaleByAdadeltaState(NamedTuple):
    """State for the Adadelta algorithm."""

    mu: Updates
    nu: Updates


def scale_by_adadelta(
    rho: float = 0.9,
    eps: float = 1e-6,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Rescale updates according to the Adadelta algorithm.

    References:
        - Zeiler, 2012: https://arxiv.org/abs/1212.5701

    Args:
        rho (float, optional): Decay rate for the squared grads.
            (default: :const:`0.9`)
        eps (float, optional): Term added to the denominator to improve numerical stability.
            (default: :const:`1e-6`)
        moment_requires_grad (bool, optional): If :data:`True`, states will be created with flag
            ``requires_grad = True``. (default: :data:`False`)

    Returns:
        An (init_fn, update_fn) tuple.
    """
    return _scale_by_adadelta(
        rho=rho,
        eps=eps,
        moment_requires_grad=moment_requires_grad,
        already_flattened=False,
    )


def _scale_by_adadelta_flat(
    rho: float = 0.9,
    eps: float = 1e-6,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    return _scale_by_adadelta(
        rho=rho,
        eps=eps,
        moment_requires_grad=moment_requires_grad,
        already_flattened=True,
    )


def _scale_by_adadelta(
    rho: float = 0.9,
    eps: float = 1e-6,
    moment_requires_grad: bool = False,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    # pylint: disable=unneeded-not
    if not eps >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= rho < 1.0:  # pragma: no cover
        raise ValueError(f'Invalid rho parameter at index 0: {rho}')
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
        return ScaleByAdadeltaState(mu=mu, nu=nu)

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
            rho,
            order=2,
            inplace=inplace,
            already_flattened=already_flattened,
        )

        if inplace:

            def f(m: torch.Tensor, v: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                return g.mul_(v.add(eps).div_(m.add(eps)).sqrt_()) if g is not None else g

        else:

            def f(m: torch.Tensor, v: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                return g.mul(v.add(eps).div_(m.add(eps)).sqrt_()) if g is not None else g

        updates = tree_map(f, mu, state.nu, updates)

        nu = update_moment.impl(  # type: ignore[attr-defined]
            updates,
            state.nu,
            rho,
            order=2,
            inplace=inplace,
            already_flattened=already_flattened,
        )

        return updates, ScaleByAdadeltaState(mu=mu, nu=nu)

    return GradientTransformation(init_fn, update_fn)


scale_by_adadelta.flat = _scale_by_adadelta_flat  # type: ignore[attr-defined]
scale_by_adadelta.impl = _scale_by_adadelta  # type: ignore[attr-defined]
