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
"""Preset transformations for scaling updates by Adam."""

# pylint: disable=invalid-name

from typing import NamedTuple

import torch

from torchopt import pytree
from torchopt.base import GradientTransformation, identity
from torchopt.transform.utils import tree_map_flat
from torchopt.typing import Params


__all__ = ['trace']


class TraceState(NamedTuple):
    """Holds an aggregation of past updates."""

    trace: Params


def trace(
    momentum: float = 0.9,
    dampening: float = 0.0,
    nesterov: bool = False,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
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
            If :data:`True`, states will be created with flag `requires_grad = True`.

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


def _trace_flat(
    momentum: float = 0.9,
    dampening: float = 0.0,
    nesterov: bool = False,
    moment_requires_grad: bool = False,
) -> GradientTransformation:
    return _trace(
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
        moment_requires_grad=moment_requires_grad,
        already_flattened=True,
    )


def _trace(
    momentum: float = 0.9,
    dampening: float = 0.0,
    nesterov: bool = False,
    moment_requires_grad: bool = False,
    *,
    already_flattened: bool = False,
) -> GradientTransformation:
    # pylint: disable=unneeded-not
    if not 0.0 <= momentum:
        raise ValueError(f'Invalid momentum value: {momentum}')
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        raise ValueError('Nesterov momentum requires a momentum and zero dampening')
    # pylint: enable=unneeded-not

    if momentum == 0.0:
        return identity()

    if already_flattened:
        tree_map = tree_map_flat
    else:
        tree_map = pytree.tree_map  # type: ignore[assignment]

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

    return GradientTransformation(init_fn, update_fn)


trace.flat = _trace_flat  # type: ignore[attr-defined]
trace.impl = _trace  # type: ignore[attr-defined]
