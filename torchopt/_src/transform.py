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

from typing import Any, Callable, List, NamedTuple, Sequence

import torch

from torchopt._src import base
from torchopt._src.typing import Schedule
from torchopt._src.utils import pytree


ScaleState = base.EmptyState
INT32_MAX = torch.iinfo(torch.int32).max
TRIPLE_PYTREEDEF = pytree.tree_structure((0, 1, 2))


def map_flattened(func: Callable, *args: Any) -> List[Any]:
    """Apply a function to each element of a flattened list."""
    return list(map(func, *args))


def with_flattened_tree(inner: base.GradientTransformation) -> base.GradientTransformation:
    # pylint: disable-next=line-too-long
    """Wraps around the inner transformation that manipulates the flattened tree structure (:class:``list``)."""

    def init_fn(params):
        return inner.init(pytree.tree_leaves(params))

    def update_fn(updates, state, *, params=None, inplace=True):
        flattened_updates, treedef = pytree.tree_flatten(updates)
        if params is not None:
            params = pytree.tree_leaves(params)

        flattened_updates, state = inner.update(
            flattened_updates, state, params=params, inplace=inplace
        )
        updates = pytree.tree_unflatten(treedef, flattened_updates)

        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


def inc_count(updates: base.Updates, count: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    """Increments int counter by one.

    Returns:
        A counter incremeted by one, or max_int if the maximum precision is reached.
    """
    return _inc_count(updates=updates, count=count, already_flattened=False)


def _inc_count(
    updates: base.Updates, count: Sequence[torch.Tensor], *, already_flattened: bool = False
) -> Sequence[torch.Tensor]:
    def f(c, g):
        return c + (c != INT32_MAX).to(torch.int32) if g is not None else c

    if already_flattened:
        return map_flattened(f, count, updates)
    return pytree.tree_map(f, count, updates)


def scale(step_size: float) -> base.GradientTransformation:
    """Scale updates by some fixed scalar ``step_size``.

    Args:
        step_size: A scalar corresponding to a fixed scaling factor for updates.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """
    return _scale(step_size=step_size, already_flattened=False)


def _scale(step_size: float, *, already_flattened: bool = False) -> base.GradientTransformation:
    if already_flattened:
        tree_map = map_flattened
    else:
        tree_map = pytree.tree_map

    def init_fn(_):
        return ScaleState()

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        if inplace:

            def f(g):
                return g.mul_(step_size) if g is not None else None

        else:

            def f(g):
                return g.mul(step_size) if g is not None else None

        updates = tree_map(f, updates)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)


class ScaleByScheduleState(NamedTuple):
    """Maintains count for scale scheduling."""

    count: Sequence[torch.Tensor]  # type: ignore


def scale_by_schedule(step_size_fn: Schedule) -> base.GradientTransformation:
    """Scale updates using a custom schedule for the ``step_size``.

    Args:
        step_size_fn:
            A function that takes an update count as input and proposes the ``step_size`` to
            multiply the updates by.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """
    return _scale_by_schedule(step_size_fn=step_size_fn, already_flattened=False)


def _scale_by_schedule(
    step_size_fn: Schedule, *, already_flattened: bool = False
) -> base.GradientTransformation:
    if already_flattened:
        tree_map = map_flattened
    else:
        tree_map = pytree.tree_map

    def init_fn(params):
        zero = tree_map(  # count init
            lambda t: torch.zeros(1, dtype=torch.int32, device=t.device), params
        )
        return ScaleByScheduleState(count=zero)

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        step_size = step_size_fn(state.count)

        if inplace:

            def f(g):
                return g.mul_(step_size) if g is not None else None

        else:

            def f(g):
                return g.mul(step_size) if g is not None else None

        updates = tree_map(f, updates)
        return updates, ScaleByScheduleState(count=inc_count(updates, state.count))

    return base.GradientTransformation(init_fn, update_fn)


def _update_moment(updates, moments, decay, *, order, inplace=True, already_flattened=False):
    """Compute the exponential moving average of the ``order``-th moment."""
    assert order in (1, 2)

    if inplace:

        if order == 2:

            def f(g, t):
                return t.mul_(decay).addcmul_(g, g, value=1 - decay) if g is not None else t

        else:

            def f(g, t):
                return t.mul_(decay).add_(g, alpha=1 - decay) if g is not None else t

    else:

        if order == 2:

            def f(g, t):
                return t.mul(decay).addcmul_(g, g, value=1 - decay) if g is not None else t

        else:

            def f(g, t):
                return t.mul(decay).add_(g, alpha=1 - decay) if g is not None else t

    if already_flattened:
        return map_flattened(f, updates, moments)
    return pytree.tree_map(f, updates, moments)


class ScaleByAdamState(NamedTuple):
    """State for the Adam algorithm."""

    mu: base.Updates
    nu: base.Updates
    count: Sequence[torch.Tensor]  # type: ignore


def _bias_correction(moment, decay, count, *, already_flattened=False):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""

    def f(t, c):
        return t.div(1 - decay**c)

    if already_flattened:
        return map_flattened(f, moment, count)
    return pytree.tree_map(f, moment, count)


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
        b1: (default: :const:`0.9`)
            Decay rate for the exponentially weighted average of grads.
        b2: (default: :const:`0.999`)
            Decay rate for the exponentially weighted average of squared grads.
        eps: (default: :const:`1e-8`)
            Term added to the denominator to improve numerical stability.
        eps_root: (default: :const:`0.0`)
            Term added to the denominator inside the square-root to improve
            numerical stability when back-propagating gradients through the rescaling.
        moment_requires_grad: (default: :data:`False`)
            if :data:`True`, states will be created with flag `requires_grad = True`.

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


def _scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
    *,
    already_flattened: bool = False,
) -> base.GradientTransformation:
    # pylint: disable=unneeded-not
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= b1 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 0: {b1}')
    if not 0.0 <= b2 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 1: {b2}')
    # pylint: enable=unneeded-not

    if already_flattened:
        tree_map = map_flattened
    else:
        tree_map = pytree.tree_map

    def init_fn(params):
        zero = tree_map(  # count init
            lambda t: torch.zeros(1, dtype=torch.int32, device=t.device), params
        )
        mu = tree_map(  # first moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        nu = tree_map(  # second moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        return ScaleByAdamState(mu=mu, nu=nu, count=zero)

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        mu = _update_moment(
            updates, state.mu, b1, order=1, inplace=inplace, already_flattened=already_flattened
        )
        nu = _update_moment(
            updates, state.nu, b2, order=2, inplace=inplace, already_flattened=already_flattened
        )
        count_inc = _inc_count(updates, state.count, already_flattened=already_flattened)
        mu_hat = _bias_correction(mu, b1, count_inc, already_flattened=already_flattened)
        nu_hat = _bias_correction(nu, b2, count_inc, already_flattened=already_flattened)

        if inplace:

            def f(g, m, v):
                return m.div_(v.add_(eps_root).sqrt_().add_(eps)) if g is not None else None

        else:

            def f(g, m, v):
                return m.div(v.add(eps_root).sqrt_().add_(eps)) if g is not None else None

        updates = tree_map(f, updates, mu_hat, nu_hat)
        return updates, ScaleByAdamState(mu=mu, nu=nu, count=count_inc)

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
        b1: (default: :const:`0.9`)
            Decay rate for the exponentially weighted average of grads.
        b2: (default: :const:`0.999`)
            Decay rate for the exponentially weighted average of squared grads.
        eps: (default: :const:`1e-8`)
            Term added to the denominator to improve numerical stability.
        eps_root: (default: :const:`0.0`)
            Term added to the denominator inside the square-root to improve
            numerical stability when back-propagating gradients through the rescaling.
        moment_requires_grad: (default: :data:`False`)
            if :data:`True`, states will be created with flag `requires_grad = True`.

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


def _scale_by_accelerated_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
    *,
    already_flattened: bool = False,
) -> base.GradientTransformation:
    # pylint: disable=unneeded-not
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= b1 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 0: {b1}')
    if not 0.0 <= b2 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 1: {b2}')
    # pylint: enable=unneeded-not

    from torchopt._src.accelerated_op import AdamOp  # pylint: disable=import-outside-toplevel

    if already_flattened:
        tree_map = map_flattened

        # pylint: disable-next=unused-argument
        def update_fn(updates, state, *, params=None, inplace=True):
            count_inc = _inc_count(updates, state.count, already_flattened=True)

            op = AdamOp(b1=b1, b2=b2, eps=eps, eps_root=eps_root, inplace=inplace)
            out = map_flattened(op, state.mu, state.nu, updates, count_inc)

            new_mu, new_nu, new_updates = tuple(zip(*out))  # transpose
            return new_updates, ScaleByAdamState(mu=new_mu, nu=new_nu, count=count_inc)

    else:
        tree_map = pytree.tree_map

        # pylint: disable-next=unused-argument
        def update_fn(updates, state, *, params=None, inplace=True):
            count_inc = _inc_count(updates, state.count, already_flattened=False)

            treedef = pytree.tree_structure(updates)

            op = AdamOp(b1=b1, b2=b2, eps=eps, eps_root=eps_root, inplace=inplace)
            out = pytree.tree_map(op, state.mu, state.nu, updates, count_inc)

            new_mu, new_nu, new_updates = pytree.tree_transpose(treedef, TRIPLE_PYTREEDEF, out)
            return new_updates, ScaleByAdamState(mu=new_mu, nu=new_nu, count=count_inc)

    def init_fn(params):
        zero = tree_map(  # count init
            lambda t: torch.zeros(1, dtype=torch.int32, device=t.device), params
        )
        mu = tree_map(  # first moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        nu = tree_map(  # second moment
            lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
        )
        return ScaleByAdamState(mu=mu, nu=nu, count=zero)

    return base.GradientTransformation(init_fn, update_fn)


class TraceState(NamedTuple):
    """Holds an aggregation of past updates."""

    trace: base.Params


def trace(
    momentum: float = 0.9,
    dampening: float = 0.0,
    nesterov: bool = False,
    moment_requires_grad: bool = False,
) -> base.GradientTransformation:
    """Compute a trace of past updates.

    Note: `trace` and `ema` have very similar but distinct updates;
    `trace = decay * trace + t`, while `ema = decay * ema + (1 - decay) * t`.
    Both are frequently found in the optimization literature.

    Args:
        momentum: (default: :const:`0.9`)
            The decay rate for the trace of past updates.
        dampening: (default: :const:`0.0`)
            Dampening for momentum.
        nesterov: (default: :data:`False`)
            Whether to use Nesterov momentum.
        moment_requires_grad: (default: :data:`False`)
            if :data:`True`, states will be created with flag `requires_grad = True`.

    Returns:
        An (init_fn, update_fn) tuple.
    """
    return _trace(
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
        moment_requires_grad=moment_requires_grad,
        already_flattened=False,
    )


def _trace(
    momentum: float = 0.9,
    dampening: float = 0.0,
    nesterov: bool = False,
    moment_requires_grad: bool = False,
    *,
    already_flattened: bool = False,
) -> base.GradientTransformation:
    # pylint: disable=unneeded-not
    if not 0.0 <= momentum:
        raise ValueError(f'Invalid momentum value: {momentum}')
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        raise ValueError('Nesterov momentum requires a momentum and zero dampening')
    # pylint: enable=unneeded-not

    if momentum == 0.0:
        return base.identity()

    if already_flattened:
        tree_map = map_flattened
    else:
        tree_map = pytree.tree_map

    def init_fn(params):
        return TraceState(
            trace=tree_map(
                lambda t: torch.zeros_like(t, requires_grad=moment_requires_grad), params
            )
        )

    first_call = True

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        nonlocal first_call

        if nesterov:
            if inplace:

                def f1(g, t):
                    if first_call:
                        return t.add_(g)
                    return t.mul_(momentum).add_(g)

                def f2(g, t):
                    return g.add_(t, alpha=momentum)

                new_trace = tree_map(f1, updates, state.trace)
                updates = tree_map(f2, updates, new_trace)
            else:

                def f1(g, t):
                    if first_call:
                        return t.add(g)
                    return t.mul(momentum).add_(g)

                def f2(g, t):
                    return g.add(t, alpha=momentum)

                new_trace = tree_map(f1, updates, state.trace)
                updates = tree_map(f2, updates, new_trace)
        else:
            if inplace:

                def f(g, t):
                    if first_call:
                        return t.add(g)
                    return t.mul_(momentum).add_(g, alpha=1.0 - dampening)

                def copy_(g, t):
                    return g.copy_(t)

                new_trace = tree_map(f, updates, state.trace)
                updates = tree_map(copy_, updates, new_trace)
            else:

                def f(g, t):
                    if first_call:
                        return t.add(g)
                    return t.mul(momentum).add_(g, alpha=1.0 - dampening)

                new_trace = tree_map(f, updates, state.trace)
                updates = tree_map(torch.clone, new_trace)

        first_call = False
        return updates, TraceState(trace=new_trace)

    return base.GradientTransformation(init_fn, update_fn)


class ScaleByRmsState(NamedTuple):
    """State for exponential root mean-squared (RMS)-normalized updates."""

    nu: base.Updates


def scale_by_rms(
    alpha: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.0
) -> base.GradientTransformation:
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


def _scale_by_rms(
    alpha: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    *,
    already_flattened: bool = False,
) -> base.GradientTransformation:
    # pylint: disable=unneeded-not
    if not 0.0 <= alpha:
        raise ValueError(f'Invalid alpha value: {alpha}')
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    # pylint: enable=unneeded-not

    if already_flattened:
        tree_map = map_flattened
    else:
        tree_map = pytree.tree_map

    def init_fn(params):
        nu = tree_map(lambda n: torch.full_like(n, initial_scale), params)  # second moment
        return ScaleByRmsState(nu=nu)

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        nu = _update_moment(
            updates, state.nu, alpha, order=2, inplace=inplace, already_flattened=already_flattened
        )

        if inplace:

            def f(g, n):
                return g.div_(n.sqrt().add_(eps))

        else:

            def f(g, n):
                return g.div(n.sqrt().add_(eps))

        updates = tree_map(f, updates, nu)
        return updates, ScaleByRmsState(nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


class ScaleByRStdDevState(NamedTuple):
    """State for centered exponential moving average of squares of updates."""

    mu: base.Updates
    nu: base.Updates


def scale_by_stddev(
    alpha: float = 0.9, eps: float = 1e-8, initial_scale: float = 0.0
) -> base.GradientTransformation:
    """Rescale updates by the root of the centered exp. moving average of squares.

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
    return _scale_by_stddev(
        alpha=alpha,
        eps=eps,
        initial_scale=initial_scale,
        already_flattened=False,
    )


def _scale_by_stddev(
    alpha: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    *,
    already_flattened: bool = False,
) -> base.GradientTransformation:
    # pylint: disable=unneeded-not
    if not 0.0 <= alpha:
        raise ValueError(f'Invalid alpha value: {alpha}')
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    # pylint: enable=unneeded-not

    if already_flattened:
        tree_map = map_flattened
    else:
        tree_map = pytree.tree_map

    def init_fn(params):
        mu = tree_map(torch.zeros_like, params)  # first moment
        nu = tree_map(lambda n: torch.full_like(n, initial_scale), params)  # second moment
        return ScaleByRStdDevState(mu=mu, nu=nu)

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        mu = _update_moment(
            updates, state.mu, alpha, order=1, inplace=inplace, already_flattened=already_flattened
        )
        nu = _update_moment(
            updates, state.nu, alpha, order=2, inplace=inplace, already_flattened=already_flattened
        )

        if inplace:

            def f(g, m, n):
                return g.div_(n.addcmul(m, m, value=-1.0).sqrt_().add_(eps))

        else:

            def f(g, m, n):
                return g.div(n.addcmul(m, m, value=-1.0).sqrt_().add_(eps))

        updates = tree_map(f, updates, mu, nu)
        return updates, ScaleByRStdDevState(mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)
