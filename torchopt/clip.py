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
# https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/clip_grad.py
# ==============================================================================
"""Utilities for gradient clipping."""

from __future__ import annotations

import torch

from torchopt import pytree
from torchopt.base import EmptyState, GradientTransformation
from torchopt.typing import OptState, Params, Updates


__all__ = ['clip_grad_norm']


ClipState = EmptyState


def clip_grad_norm(
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> GradientTransformation:
    """Clip gradient norm of an iterable of parameters.

    Args:
        max_norm (float): The maximum absolute value for each element in the update.
        norm_type (float, optional): Type of the used p-norm. Can be ``'inf'`` for infinity norm.
            (default: :const:`2.0`)
        error_if_nonfinite (bool, optional): If :data:`True`, an error is thrown if the total norm
            of the gradients from ``updates`` is ``nan``, ``inf``, or ``-inf``.
            (default: :data:`False`)

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(params: Params) -> OptState:  # pylint: disable=unused-argument
        return ClipState()

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,  # pylint: disable=unused-argument
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        available_updates = pytree.tree_leaves(updates)
        if len(available_updates) == 0:
            return updates, state
        device = available_updates[0].device
        with torch.no_grad():
            if norm_type == torch.inf:
                norms = [p.abs().max().to(device) for p in available_updates]
                total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
            else:
                total_norm = torch.norm(
                    torch.stack([torch.norm(p, norm_type).to(device) for p in available_updates]),
                    norm_type,
                )
            if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
                raise RuntimeError(
                    f'The total norm of order {norm_type} for gradients from `parameters` is '
                    f'non-finite, so it cannot be clipped. To disable this error and scale the '
                    f'gradients by the non-finite norm anyway, set `error_if_nonfinite=False`',
                )
        clip_coefficient = max_norm / (float(total_norm) + 1e-6)
        # Note: multiplying by the clamped coefficient is redundant when the coefficient is
        # clamped to 1, but doing so avoids a `if clip_coefficient < 1:` conditional which
        # can require a CPU <=> device synchronization when the gradients do not reside in
        # CPU memory.
        clip_coefficient_clamped = min(clip_coefficient, 1.0)
        if inplace:

            def f(g: torch.Tensor) -> torch.Tensor:
                return g.mul_(clip_coefficient_clamped)

        else:

            def f(g: torch.Tensor) -> torch.Tensor:
                return g.mul(clip_coefficient_clamped)

        new_updates = pytree.tree_map(f, updates)
        return new_updates, state

    return GradientTransformation(init_fn, update_fn)
