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
# https://github.com/pytorch/pytorch/blob/master/torch/nn/utils/clip_grad.py
# ==============================================================================

import torch
from torch._six import inf

from torchopt._src import base
from torchopt._src.utils import pytree


ClipState = base.EmptyState


def clip_grad_norm(
    max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False
) -> base.GradientTransformation:
    """Clips gradient norm of an iterable of parameters.

    Args:
        max_delta: The maximum absolute value for each element in the update.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(params):  # pylint: disable=unused-argument
        return ClipState()

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        available_updates = []
        for g in updates:
            if g is not None:
                available_updates.append(g)
        if len(available_updates) == 0:
            return torch.tensor(0.0)
        device = available_updates[0].device
        with torch.no_grad():
            if norm_type == inf:
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
                    f'gradients by the non-finite norm anyway, set `error_if_nonfinite=False`'
                )
        clip_coef = max_norm / (float(total_norm) + 1e-6)
        # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but
        # doing so avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device
        # synchronization when the gradients do not reside in CPU memory.
        clip_coef_clamped = min(clip_coef, 1.0)
        if inplace:

            def f(g):
                return g.mul_(clip_coef_clamped) if g is not None else None

        else:

            def f(g):
                return g.mul(clip_coef_clamped) if g is not None else None

        new_updates = pytree.tree_map(f, updates)
        return new_updates, state

    return base.GradientTransformation(init_fn, update_fn)
