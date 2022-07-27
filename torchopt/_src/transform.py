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

from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import torch

from torchopt._src import base
from torchopt._src.typing import Schedule, TensorTree


ScaleState = base.EmptyState


def inc_count(updates, count: Tuple[int]) -> Tuple[int]:
    """Increments int counter by one."""

    def f(c, g):
        return c + 1 if g is not None else c

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

    count: Tuple[int, ...]  # type: ignore


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
        return ScaleByScheduleState(count=tuple(0 for _ in range(len(params))))

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

    count: Tuple[int, ...]  # type: ignore
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
        mu = jax.tree_map(  # First moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        nu = jax.tree_map(  # Second moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        return ScaleByAdamState(count=tuple(0 for _ in range(len(mu))), mu=tuple(mu), nu=tuple(nu))

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
        mu = jax.tree_map(  # First moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        nu = jax.tree_map(  # Second moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        return ScaleByAdamState(count=tuple(0 for _ in range(len(params))), mu=mu, nu=nu)

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


class MaskedState(NamedTuple):
    """Maintains inner transform state for masked transformations."""

    inner_state: Any


class MaskedNode(NamedTuple):
    """A node used to mask out unspecified parts of a tree.
    This node is ignored when mapping functions across the tree e.g. using
    `jax.tree_util.tree_map` since it is a container without children. It can
    therefore be used to mask out parts of a tree.
    """


def masked(
    inner: base.GradientTransformation,
    mask: Union[TensorTree, Callable[[base.Params], TensorTree]],
) -> base.GradientTransformation:
    """Mask updates so only some are transformed, the rest are passed through.
    For example, it is common to skip weight decay for BatchNorm scale and all
    bias parameters. In many networks, these are the only parameters with only
    one dimension. So, you may create a mask function to mask these out as
    follows::
      mask_fn = lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
      weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask_fn)
    You may alternatively create the mask pytree upfront::
      mask = jax.tree_util.tree_map(lambda x: x.ndim != 1, params)
      weight_decay = optax.masked(optax.add_decayed_weights(0.001), mask)
    For the ``inner`` transform, state will only be stored for the parameters that
    have a mask value of ``True``.
    Args:
      inner: Inner transformation to mask.
      mask: a PyTree with same structure as (or a prefix of) the params PyTree, or
        a Callable that returns such a pytree given the params/updates. The leaves
        should be booleans, ``True`` for leaves/subtrees you want to apply the
        transformation to, and ``False`` for those you want to skip. The mask must
        be static for the gradient transformation to be jit-compilable.
    Returns:
      New GradientTransformation wrapping ``inner``.
    """

    def mask_pytree(pytree, mask_tree):
        return jax.tree_map(lambda m, p: p if m else MaskedNode(), mask_tree, pytree)

    def init_fn(params):
        mask_tree = mask(params) if callable(mask) else mask
        masked_params = mask_pytree(params, mask_tree)
        return MaskedState(inner_state=inner.init(masked_params))

    def update_fn(updates, state, params=None):
        mask_tree = mask(updates) if callable(mask) else mask
        masked_updates = mask_pytree(updates, mask_tree)
        masked_params = None if params is None else mask_pytree(params, mask_tree)

        new_masked_updates, new_inner_state = inner.update(
            masked_updates, state.inner_state, masked_params
        )

        new_updates = jax.tree_map(
            lambda m, new_u, old_u: new_u if m else old_u, mask_tree, new_masked_updates, updates
        )
        return new_updates, MaskedState(inner_state=new_inner_state)

    return base.GradientTransformation(init_fn, update_fn)


AddDecayedWeightsState = base.EmptyState


def add_decayed_weights(
    weight_decay: float = 0.0, mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None
) -> base.GradientTransformation:
    """Add parameter scaled by `weight_decay`.

    Args:
        weight_decay: a scalar weight decay rate.
        mask: a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the transformation to, and `False` for those you want to skip.

    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        del params
        return AddDecayedWeightsState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(
                (
                    'You are using a transformation that requires the current value of '
                    'parameters, but you are not passing `params` when calling `update`.'
                )
            )
        updates = jax.tree_map(lambda g, p: g + weight_decay * p, updates, params)
        return updates, state

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return masked(base.GradientTransformation(init_fn, update_fn), mask)
    return base.GradientTransformation(init_fn, update_fn)
