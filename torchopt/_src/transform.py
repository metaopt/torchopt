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

# pylint: disable=invalid-name

from typing import NamedTuple, Tuple

import jax
import torch

from torchopt._src import base
from torchopt._src.typing import Schedule


ScaleState = base.EmptyState


def inc_count(updates, count: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Increments int counter by one.

    Returns:
        A counter incremeted by one, or max_int if the maximum precision is reached.
    """
    max_int32_value = torch.iinfo(torch.int32).max
    one = torch.ones(1, dtype=torch.int32, device=count[0].device)

    def f(c, g):
        return (
            c + (1 - torch.div(c, max_int32_value, rounding_mode='trunc')) * one
            if g is not None
            else c
        )

    return jax.tree_map(f, count, updates)


def scale(step_size: float) -> base.GradientTransformation:
    """Scale updates by some fixed scalar ``step_size``.

    Args:
        step_size: A scalar corresponding to a fixed scaling factor for updates.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(params):
        del params
        return ScaleState()

    def update_fn(updates, state, inplace=True):
        if inplace:

            def f(g):
                return g.mul_(step_size) if g is not None else None

        else:

            def f(g):
                return g.mul(step_size) if g is not None else None

        updates = jax.tree_map(f, updates)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


class ScaleByScheduleState(NamedTuple):
    """Maintains count for scale scheduling."""

    count: Tuple[torch.Tensor, ...]  # type: ignore


def scale_by_schedule(step_size_fn: Schedule) -> base.GradientTransformation:
    """Scale updates using a custom schedule for the ``step_size``.

    Args:
        step_size_fn:
            A function that takes an update count as input and proposes the ``step_size`` to
            multiply the updates by.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(params):
        zero = jax.tree_map(  # First moment
            lambda t: torch.zeros(1, dtype=torch.int32, device=t.device), params
        )
        return ScaleByScheduleState(count=tuple(zero))

    def update_fn(updates, state, inplace=True):
        step_size = step_size_fn(state.count)
        if inplace:
            updates = jax.tree_map(lambda g, step_size: g.mul_(step_size), updates, step_size)
        else:
            updates = jax.tree_map(lambda g, step_size: g.mul(step_size), updates, step_size)
        return updates, ScaleByScheduleState(count=inc_count(updates, state.count))

    return base.GradientTransformation(init_fn, update_fn)


def _update_moment(updates, moments, decay, order, inplace=True):
    """Compute the exponential moving average of the ``order``-th moment."""
    if inplace:

        def f(g, t):
            return t.mul_(decay).add_(g**order, alpha=1 - decay) if g is not None else t

    else:

        def f(g, t):
            return t.mul(decay).add(g**order, alpha=1 - decay) if g is not None else t

    return jax.tree_map(f, updates, moments)


def _update_moment_per_elem_norm(updates, moments, decay, order, inplace=True):
    """Compute the EMA of the `order`-th moment of the element-wise norm."""
    if inplace:

        def f(g, t):
            return t.mul_(decay).add_(g**order, alpha=1 - decay) if g is not None else t

    else:

        def f(g, t):
            return t.mul(decay).add(g**order, alpha=1 - decay) if g is not None else t

    return jax.tree_map(f, updates, moments)


class ScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""

    count: Tuple[torch.Tensor, ...]  # type: ignore
    mu: base.Updates
    nu: base.Updates


def _bias_correction(moment, decay, count, inplace=True):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    if inplace:

        def f(t, c):
            return t.div_(1 - decay**c)

    else:

        def f(t, c):
            return t.div(1 - decay**c)

    return jax.tree_map(f, moment, count)


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
) -> base.GradientTransformation:
    """Rescale updates according to the Adam algorithm.

    References:
        [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

    Args:
        b1:
            Decay rate for the exponentially weighted average of grads.
        b2:
            Decay rate for the exponentially weighted average of squared grads.
        eps:
            Term added to the denominator to improve numerical stability.
        eps_root:
            Term added to the denominator inside the square-root to improve
            numerical stability when back-propagating gradients through the rescaling.
        moment_requires_grad:
            If true, states will be created with flag `requires_grad = True`.

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        zero = jax.tree_map(  # First moment
            lambda t: torch.zeros(1, dtype=torch.int32, device=t.device), params
        )
        mu = jax.tree_map(  # First moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        nu = jax.tree_map(  # Second moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        return ScaleByAdamState(count=tuple(zero), mu=tuple(mu), nu=tuple(nu))

    def update_fn(updates, state, inplace=True):
        mu = _update_moment(updates, state.mu, b1, 1, inplace)
        nu = _update_moment_per_elem_norm(updates, state.nu, b2, 2, inplace)
        count_inc = inc_count(updates, state.count)
        mu_hat = _bias_correction(mu, b1, count_inc, False)
        nu_hat = _bias_correction(nu, b2, count_inc, False)
        if inplace:

            def f(g, m, v):
                return m.div_(torch.sqrt_(v.add_(eps_root)).add_(eps)) if g is not None else None

        else:

            def f(g, m, v):
                return m.div(torch.sqrt(v.add(eps_root)).add(eps)) if g is not None else None

        updates = jax.tree_map(f, updates, mu_hat, nu_hat)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_accelerated_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
) -> base.GradientTransformation:
    """Rescale updates according to the Adam algorithm.

    This function is accelerated by using some fused accelerated operators.

    References:
        [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

    Args:
        b1:
            Decay rate for the exponentially weighted average of grads.
        b2:
            Decay rate for the exponentially weighted average of squared grads.
        eps:
            Term added to the denominator to improve numerical stability.
        eps_root:
            Term added to the denominator inside the square-root to improve
            numerical stability when back-propagating gradients through the rescaling.
        moment_requires_grad:
            If true, states will be created with flag `requires_grad = True`.

    Returns:
        An (init_fn, update_fn) tuple.
    """
    from torchopt._src.accelerated_op import AdamOp  # pylint: disable=import-outside-toplevel

    def init_fn(params):
        zero = jax.tree_map(  # First moment
            lambda t: torch.zeros(1, dtype=torch.int32, device=t.device), params
        )
        mu = jax.tree_map(  # First moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        nu = jax.tree_map(  # Second moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        return ScaleByAdamState(count=tuple(zero), mu=mu, nu=nu)

    def update_fn(updates, state, inplace=True):
        count_inc = inc_count(updates, state.count)
        op = AdamOp(b1, b2, eps, eps_root, inplace)
        out = jax.tree_map(op, state.mu, state.nu, updates, count_inc)
        new_mus, new_nus, new_updates = [], [], []
        for new_mu, new_nu, new_update in out:
            new_mus.append(new_mu)
            new_nus.append(new_nu)
            new_updates.append(new_update)
        return tuple(new_updates), ScaleByAdamState(
            count=count_inc, mu=tuple(new_mus), nu=tuple(new_nus)
        )

    return base.GradientTransformation(init_fn, update_fn)


class TraceState(NamedTuple):
    """Holds an aggregation of past updates."""

    trace: base.Params


def trace(
    decay: float,
    nesterov: bool = False,
    moment_requires_grad: bool = False,
) -> base.GradientTransformation:
    """Compute a trace of past updates.

    Note: `trace` and `ema` have very similar but distinct updates;
    `trace = decay * trace + t`, while `ema = decay * ema + (1-decay) * t`.
    Both are frequently found in the optimization literature.

    Args:
        decay:
            The decay rate for the trace of past updates.
        nesterov:
            Whether to use Nesterov momentum.
        moment_requires_grad:
            If true, states will be created with flag `requires_grad = True`.

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        if decay == 0.0:
            return TraceState(trace=())

        return TraceState(
            trace=jax.tree_map(
                lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad),
                params,
            )
        )

    def update_fn(updates, state, inplace=True):
        if nesterov:
            if inplace:

                def f1(g, t):
                    return t.copy_(g.add(t, alpha=decay))

                def f2(g, t):
                    return g.add_(t, alpha=decay)

                new_trace = jax.tree_map(f1, updates, state.trace)
                updates = jax.tree_map(f2, updates, new_trace)
            else:

                def f(g, t):
                    return g.add(t, alpha=decay)

                new_trace = jax.tree_map(f, updates, state.trace)
                updates = jax.tree_map(f, updates, new_trace)
        else:
            if inplace:

                def f(g, t):
                    return g.add_(t, alpha=decay)

                updates = jax.tree_map(f, updates, state.trace)
                state.trace.copy_(updates)
                new_trace = state.trace
            else:

                def f(g, t):
                    return g.add(t, alpha=decay)

                updates = jax.tree_map(f, updates, state.trace)
                new_trace = updates

        return updates, TraceState(trace=new_trace)

    return base.GradientTransformation(init_fn, update_fn)


class ScaleByRmsState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""

    nu: base.Updates


def scale_by_rms(
    decay: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.0
) -> base.GradientTransformation:
    """Rescale updates by the root of the exp. moving avg of the square.

    References:
        [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    Args:
        decay:
            Decay rate for the exponentially weighted average of squared grads.
        eps:
            Term added to the denominator to improve numerical stability.
        initial_scale:
            Initial value for second moment

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        nu = jax.tree_map(lambda n: torch.full_like(n, initial_scale), params)  # second moment
        return ScaleByRmsState(nu=nu)

    def update_fn(updates, state, inplace=True):
        nu = _update_moment_per_elem_norm(updates, state.nu, decay, 2, inplace)
        if inplace:

            def f(g, n):
                return g.mul_(torch.rsqrt(n.add(eps)))

        else:

            def f(g, n):
                return g.mul(torch.rsqrt(n.add(eps)))

        # """The followings are pytorch style"""
        #
        # if inplace:
        #     def f(g, n): return g.div_(torch.sqrt_(n).add_(eps))
        # else:
        #     def f(g, n): return g.div(torch.sqrt(n).add(eps))
        #
        updates = jax.tree_map(f, updates, nu)
        return updates, ScaleByRmsState(nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


class ScaleByRStdDevState(NamedTuple):
    """State for centered exponential moving average of squares of updates."""

    mu: base.Updates
    nu: base.Updates


def scale_by_stddev(
    decay: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.0
) -> base.GradientTransformation:
    """Rescale updates by the root of the centered exp. moving average of squares.

    References:
        [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

    Args:
        decay:
            Decay rate for the exponentially weighted average of squared grads.
        eps:
            Term added to the denominator to improve numerical stability.
        initial_scale:
            Initial value for second moment

    Returns:
        An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        mu = jax.tree_map(torch.zeros_like, params)  # First moment
        nu = jax.tree_map(lambda n: torch.full_like(n, initial_scale), params)  # second moment
        return ScaleByRStdDevState(mu=mu, nu=nu)

    def update_fn(updates, state, inplace=True):
        mu = _update_moment(updates, state.mu, decay, 1, inplace)
        nu = _update_moment_per_elem_norm(updates, state.nu, decay, 2, inplace)
        if inplace:

            def f(g, m, n):
                return g.mul_(torch.rsqrt(n.sub(m**2).add(eps)))

        else:

            def f(g, m, n):
                return g.mul(torch.rsqrt(n.sub(m**2).add(eps)))

        # """The followings are pytorch style"""
        #
        # if inplace:
        #     def f(g, m, n): return g.div_(torch.sqrt_(n.sub_(m ** 2)).add(eps))
        # else:
        #     def f(g, m, n): return g.div(torch.sqrt(n.sub(m ** 2)).add(eps))
        #
        updates = jax.tree_map(f, updates, mu, nu)
        return updates, ScaleByRStdDevState(mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)
