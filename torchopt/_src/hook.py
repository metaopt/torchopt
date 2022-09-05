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

import torch

from torchopt._src.base import EmptyState, GradientTransformation
from torchopt._src.utils import pytree


def zero_nan_hook(g: torch.Tensor) -> torch.Tensor:
    """Registers a zero nan hook to replace nan with zero."""
    return torch.where(torch.isnan(g), torch.zeros_like(g), g)


def register_hook(hook) -> GradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

    This function passes through the *gradient updates* unchanged.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(params):  # pylint: disable=unused-argument
        return EmptyState()

    def update_fn(updates, state, *, params=None, inplace=True):  # pylint: disable=unused-argument
        def f(g):
            return g.register_hook(hook) if g is not None else None

        pytree.tree_map(f, updates)
        return updates, state

    return GradientTransformation(init_fn, update_fn)
