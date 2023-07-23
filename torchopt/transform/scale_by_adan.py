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
"""Preset transformations for scaling updates by Adan."""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.base import GradientTransformation
from torchopt.transform.utils import inc_count, tree_map_flat, update_moment
from torchopt.typing import OptState, Params, Updates


__all__ = [
    'scale_by_adan',
]


class ScaleByAdanState(NamedTuple):
    """State for the Adan algorithm."""

    mu: Updates
    nu: Updates
    delta: Updates
    grad_tm1: Updates
    count: OptState


def _adan_bias_correction(
    moment: Updates,
    decay: float,
    count: OptState,
    *,
    already_flattened: bool = False,
) -> Updates:
    """Perform bias correction. This becomes a no-op as count goes to infinity."""

    def f(t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        return t.div(1 - pow(decay, c))

    if already_flattened:
        return tree_map_flat(f, moment, count)
    return pytree.tree_map(f, moment, count)


def scale_by_adan(
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Rescale updates according to the Adan algorithm.

    References:
        - Xie et al., 2022: https://arxiv.org/pdf/2208.06677.pdf

    Args:
        b1 (float, optional): Decay rate for the exponentially weighted average of gradients.
            (default: :const:`0.98`)
        b2 (float, optional): Decay rate for the exponentially weighted average of difference of
            gradients.
            (default: :const:`0.92`)
        b3 (float, optional): Decay rate for the exponentially weighted average of the squared term.
            (default: :const:`0.99`)
        eps (float, optional): Term added to the denominator to improve numerical stability.
            (default: :const:`1e-8`)
        eps_root (float, optional): Term added to the denominator inside the square-root to improve
            numerical stability when backpropagating gradients through the rescaling.
            (default: :const:`0.0`)
        moment_requires_grad (bool, optional): If :data:`True`, states will be created with flag
            ``requires_grad = True``. (default: :data:`False`)

    Returns:
        An (init_fn, update_fn) tuple.
    """
    return _scale_by_adan(
        b1=b1,
        b2=b2,
        b3=b3,
        eps=eps,
        eps_root=eps_root,
        moment_requires_grad=moment_requires_grad,
        already_flattened=False,
    )


def _scale_by_adan_flat(
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    return _scale_by_adan(
        b1=b1,
        b2=b2,
        b3=b3,
        eps=eps,
        eps_root=eps_root,
        moment_requires_grad=moment_requires_grad,
        already_flattened=True,
    )


def _scale_by_adan(
    b1: float = 0.98,
    b2: float = 0.92,
    b3: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    # pylint: disable=unneeded-not
    if not eps >= 0.0:  # pragma: no cover
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= b1 < 1.0:  # pragma: no cover
        raise ValueError(f'Invalid beta parameter at index 0: {b1}')
    if not 0.0 <= b2 < 1.0:  # pragma: no cover
        raise ValueError(f'Invalid beta parameter at index 1: {b2}')
    if not 0.0 <= b3 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 2: {b3}')
    # pylint: enable=unneeded-not

    if already_flattened:  # noqa: SIM108
        tree_map = tree_map_flat
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]

    def init_fn(params: Params) -> OptState:
        tree_map(  # count init
            lambda t: torch.zeros(1, dtype=torch.int64, device=t.device).squeeze_(),
            params,
        )
        mu = tree_map(  # first moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad),
            params,
        )
        nu = tree_map(  # second moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad),
            params,
        )
        delta = tree_map(  # EWA of Difference of gradients
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad),
            params,
        )
        grad_tm1 = tree_map(
            lambda t: torch.zeros_like(
                t,
            ),
            params,
        )  # Previous gradient
        return ScaleByAdanState(
            mu=mu,
            nu=nu,
            delta=delta,
            grad_tm1=grad_tm1,
            count=zero,
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,  # pylint: disable=unused-argument
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        diff = pytree.lax.cond(
            state.count != 0,
            lambda X, Y: tree_map(lambda x, y: x - y, X, Y),
            lambda X, _: tree_map(torch.zeros_like, X),
            updates,
            state.grad_tm1,
        )

        grad_prime = tree_map(lambda g, d: g + b2 * d, updates, diff)

        mu = update_moment.impl(
            updates,
            state.mu,
            b1,
            order=1,
            inplace=inplace,
            already_flattened=already_flattened,
        )
        delta = update_moment.impl(
            diff,
            state.delta,
            b2,
            1,
        )
        nu = update_moment_per_elem_norm(grad_prime, state.nu, b3, 2)

        count_inc = inc_count.impl(updates, state.count, already_flattened=already_flattened)  # type: ignore[attr-defined]
        mu_hat = _adan_bias_correction(mu, b1, count_inc, already_flattened=already_flattened)
        delta_hat = _adan_bias_correction(delta, b2, count_inc, already_flattened=already_flattened)
        nu_hat = _adan_bias_correction(nu, b3, count_inc, already_flattened=already_flattened)

        if inplace:

            def f(
                m: torch.Tensor,
                d: torch.Tensor,
                n: torch.Tensor,
            ) -> torch.Tensor:
                return (m + b2 * d).div_(torch.sqrt(n + eps_root).add(eps))

        else:

            def f(
                m: torch.Tensor,
                d: torch.Tensor,
                n: torch.Tensor,
            ) -> torch.Tensor:
                return (m + b2 * d).div(torch.sqrt(n + eps_root).add(eps))

        # lambda m, d, n: (m + b2 * d) / (torch.sqrt(n + eps_root) + eps),
        updates = pytree.tree_map(f, mu_hat, delta_hat, nu_hat)

        return updates, ScaleByAdanState(
            count=count_inc, mu=mu, nu=nu, delta=delta, grad_tm1=updates,
        )

    return GradientTransformation(init_fn, update_fn)


scale_by_adan.flat = _scale_by_adan_flat  # type: ignore[attr-defined]
scale_by_adan.impl = _scale_by_adan  # type: ignore[attr-defined]


# def scale_by_proximal_adan(
#     learning_rate: ScalarOrSchedule,
#     weight_decay: float,
#     b1: float = 0.98,
#     b2: float = 0.92,
#     b3: float = 0.99,
#     eps_root: float = 1e-8,
#     fo_dtype: Optional[Any] = None,
# ) -> base.GradientTransformation:
#   """Rescale updates according to the proximal version of the Adan algorithm.
#   References:
#     [Xie et al, 2022](https://arxiv.org/abs/2208.06677)
#   Args:
#     b1: Decay rate for the exponentially weighted average of gradients.
#     b2: Decay rate for the exponentially weighted average of difference of
#       gradients.
#     b3: Decay rate for the exponentially weighted average of the squared term.
#     eps: term added to the denominator to improve numerical stability.
#     eps_root: Term added to the denominator inside the square-root to improve
#       numerical stability when backpropagating gradients through the rescaling.
#     fo_dtype: optional `dtype` to be used for the first order accumulators
#       mu and delta; if `None` then the `dtype is inferred from `params`
#       and `updates`.
#   Returns:
#     An (init_fn, update_fn) tuple.
#   """

#   fo_dtype = utils.canonicalize_dtype(fo_dtype)

#   def init_fn(params):
#     mu = pytree.tree_map(  # First moment
#         lambda t: torch.zeros_like(t, dtype=fo_dtype), params)
#     nu = pytree.tree_map(torch.zeros_like, params)  # Second moment
#     delta = pytree.tree_map(  # EWA of Difference of gradients
#         lambda t: torch.zeros_like(t, dtype=fo_dtype), params)
#     grad_tm1 = pytree.tree_map(torch.zeros_like, params)  # Previous gradient
#     return ScaleByAdanState(count=torch.zeros([], torch.int32),
#                             mu=mu, nu=nu, delta=delta, grad_tm1=grad_tm1)

#   def update_fn(updates, state, params=None):
#     diff = pytree.lax.cond(state.count != 0,
#                         lambda X, Y: pytree.tree_map(lambda x, y: x - y, X, Y),
#                         lambda X, _: pytree.tree_map(torch.zeros_like, X),
#                         updates, state.grad_tm1)

#     grad_prime = pytree.tree_map(lambda g, d: g + b2*d, updates, diff)

#     mu = update_moment(updates, state.mu, b1, 1)
#     delta = update_moment(diff, state.delta, b2, 1)
#     nu = update_moment_per_elem_norm(grad_prime, state.nu, b3, 2)

#     count_inc = numerics.safe_int32_increment(state.count)
#     mu_hat = _adan_bias_correction(mu, b1, count_inc)
#     delta_hat = _adan_bias_correction(delta, b2, count_inc)
#     nu_hat = _adan_bias_correction(nu, b3, count_inc)

#     if callable(learning_rate):
#       lr = learning_rate(state.count)
#     else:
#       lr = learning_rate

#     learning_rates = pytree.tree_util.tree_map(
#       lambda n: lr / torch.sqrt(n + eps_root), nu_hat)

#     # negative scale: gradient descent
#     updates = pytree.tree_util.tree_map(lambda scale, m, v: -scale * (m + b2 * v),
#                                      learning_rates, mu_hat,
#                                      delta_hat)

#     decay = 1. / (1. + weight_decay * lr)
#     params_new = pytree.tree_util.tree_map(lambda p, u:
#                                         decay * (p + u), params,
#                                         updates)

#     # params_new - params_old
#     new_updates = pytree.tree_util.tree_map(lambda new, old: new - old, params_new,
#                                      params)

#     mu_hat = utils.cast_tree(mu_hat, fo_dtype)
#     delta_hat = utils.cast_tree(delta_hat, fo_dtype)

#     return new_updates, ScaleByAdanState(count=count_inc,
#                                     mu=mu, nu=nu, delta=delta,
#                                     grad_tm1=updates)

#   return GradientTransformation(init_fn, update_fn)
