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
"""Preset transformations for scaling updates by Adam."""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.accelerated_op import AdamOp
from torchopt.base import GradientTransformation
from torchopt.transform.utils import inc_count, tree_map_flat, update_moment
from torchopt.typing import OptState, Params, Updates


__all__ = ['scale_by_adam', 'scale_by_accelerated_adam']


TRIPLE_PYTREE_SPEC = pytree.tree_structure((0, 1, 2), none_is_leaf=True)  # type: ignore[arg-type]


class ScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""

    mu: Updates
    nu: Updates
    count: OptState


def _bias_correction(
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


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Rescale updates according to the Adam algorithm.

    References:
        - Kingma et al., 2014: https://arxiv.org/abs/1412.6980

    Args:
        b1 (float, optional): Decay rate for the exponentially weighted average of grads.
            (default: :const:`0.9`)
        b2 (float, optional): Decay rate for the exponentially weighted average of squared grads.
            (default: :const:`0.999`)
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
    return _scale_by_adam(
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        moment_requires_grad=moment_requires_grad,
        already_flattened=False,
    )


def _scale_by_adam_flat(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    return _scale_by_adam(
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        moment_requires_grad=moment_requires_grad,
        already_flattened=True,
    )


# pylint: disable-next=too-many-arguments
def _scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
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
    # pylint: enable=unneeded-not

    if already_flattened:  # noqa: SIM108
        tree_map = tree_map_flat
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]

    def init_fn(params: Params) -> OptState:
        zero = tree_map(  # count init
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
        return ScaleByAdamState(mu=mu, nu=nu, count=zero)

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
        # pylint: disable=line-too-long
        count_inc = inc_count.impl(updates, state.count, already_flattened=already_flattened)  # type: ignore[attr-defined]
        mu_hat = _bias_correction(mu, b1, count_inc, already_flattened=already_flattened)
        nu_hat = _bias_correction(nu, b2, count_inc, already_flattened=already_flattened)

        if inplace:

            def f(m: torch.Tensor, v: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                return m.div_(v.add_(eps_root).sqrt_().add(eps)) if g is not None else g

        else:

            def f(m: torch.Tensor, v: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor | None:
                return m.div(v.add(eps_root).sqrt_().add(eps)) if g is not None else g

        updates = tree_map(f, mu_hat, nu_hat, updates)
        return updates, ScaleByAdamState(mu=mu, nu=nu, count=count_inc)

    return GradientTransformation(init_fn, update_fn)


scale_by_adam.flat = _scale_by_adam_flat  # type: ignore[attr-defined]
scale_by_adam.impl = _scale_by_adam  # type: ignore[attr-defined]


def scale_by_accelerated_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    """Rescale updates according to the Adam algorithm.

    This function is accelerated by using some fused accelerated operators.

    References:
        - Kingma et al., 2014: https://arxiv.org/abs/1412.6980

    Args:
        b1 (float, optional): Decay rate for the exponentially weighted average of grads.
            (default: :const:`0.9`)
        b2 (float, optional): Decay rate for the exponentially weighted average of squared grads.
            (default: :const:`0.999`)
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
    return _scale_by_accelerated_adam(
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        moment_requires_grad=moment_requires_grad,
        already_flattened=False,
    )


def _scale_by_accelerated_adam_flat(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    return _scale_by_accelerated_adam(
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        moment_requires_grad=moment_requires_grad,
        already_flattened=True,
    )


# pylint: disable-next=too-many-arguments
def _scale_by_accelerated_adam(
    b1: float = 0.9,
    b2: float = 0.999,
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
    # pylint: enable=unneeded-not

    if already_flattened:
        tree_map = tree_map_flat

        def update_fn(
            updates: Updates,
            state: OptState,
            *,
            params: Params | None = None,  # pylint: disable=unused-argument
            inplace: bool = True,
        ) -> tuple[Updates, OptState]:
            count_inc = inc_count.impl(updates, state.count, already_flattened=True)  # type: ignore[attr-defined]

            op = AdamOp(b1=b1, b2=b2, eps=eps, eps_root=eps_root, inplace=inplace)

            def op_fn(
                mu: torch.Tensor | None,
                nu: torch.Tensor | None,
                update: torch.Tensor | None,
                count: torch.Tensor | None,
            ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
                if mu is None:
                    return (None, None, None)
                return op(mu, nu, update, count)  # type: ignore[arg-type]

            out = tree_map_flat(
                op_fn,
                state.mu,
                state.nu,
                updates,
                count_inc,
                none_is_leaf=True,
            )

            if len(out) == 0:
                new_mu, new_nu, new_updates = (), (), ()
            else:
                new_mu, new_nu, new_updates = tuple(zip(*out))  # transpose

            new_mu, new_nu, new_updates = (
                new if type(new) is type(old) else type(old)(new)
                for new, old in (
                    (new_mu, state.mu),
                    (new_nu, state.nu),
                    (new_updates, updates),
                )
            )

            return new_updates, ScaleByAdamState(mu=new_mu, nu=new_nu, count=count_inc)

    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]

        def update_fn(
            updates: Updates,
            state: OptState,
            *,
            params: Params | None = None,  # pylint: disable=unused-argument
            inplace: bool = True,
        ) -> tuple[Updates, OptState]:
            count_inc = inc_count.impl(updates, state.count, already_flattened=False)  # type: ignore[attr-defined]

            new_mu: Updates
            new_nu: Updates
            new_updates: Updates

            treespec = pytree.tree_structure(updates, none_is_leaf=True)
            if treespec.num_leaves > 0:
                op = AdamOp(b1=b1, b2=b2, eps=eps, eps_root=eps_root, inplace=inplace)

                def op_fn(
                    mu: torch.Tensor | None,
                    nu: torch.Tensor | None,
                    update: torch.Tensor | None,
                    count: torch.Tensor | None,
                ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
                    if mu is None:
                        return (None, None, None)
                    return op(mu, nu, update, count)  # type: ignore[arg-type]

                out = pytree.tree_map(
                    op_fn,
                    state.mu,
                    state.nu,
                    updates,
                    count_inc,
                    none_is_leaf=True,
                )

                new_mu, new_nu, new_updates = pytree.tree_transpose(  # type: ignore[misc]
                    treespec,
                    TRIPLE_PYTREE_SPEC,
                    out,
                )
            else:
                new_mu = pytree.tree_unflatten(treespec, ())
                new_nu = pytree.tree_unflatten(treespec, ())
                new_updates = pytree.tree_unflatten(treespec, ())

            return new_updates, ScaleByAdamState(mu=new_mu, nu=new_nu, count=count_inc)

    def init_fn(params: Params) -> OptState:
        zero = tree_map(  # count init
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
        return ScaleByAdamState(mu=mu, nu=nu, count=zero)

    return GradientTransformation(init_fn, update_fn)


scale_by_accelerated_adam.flat = _scale_by_accelerated_adam_flat  # type: ignore[attr-defined]
scale_by_accelerated_adam.impl = _scale_by_accelerated_adam  # type: ignore[attr-defined]
