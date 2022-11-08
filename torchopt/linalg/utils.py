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
"""Utilities for linear algebra."""

import itertools
from functools import partial
from typing import Callable, Tuple, Union

import torch

from torchopt import pytree
from torchopt.typing import TensorTree


def cat_shapes(tree: TensorTree) -> Tuple[int, ...]:
    """Concatenate the shapes of the leaves of a tree of tensors."""
    leaves = pytree.tree_leaves(tree)
    return tuple(itertools.chain.from_iterable(tuple(leaf.shape) for leaf in leaves))


def normalize_matvec(
    matvec: Union[Callable[[TensorTree], TensorTree], torch.Tensor]
) -> Callable[[TensorTree], TensorTree]:
    """Normalize an argument for computing matrix-vector products."""
    if callable(matvec):
        return matvec

    assert isinstance(matvec, torch.Tensor)
    if matvec.ndim != 2 or matvec.shape[0] != matvec.shape[1]:
        raise ValueError(f'linear operator must be a square matrix, but has shape: {matvec.shape}')
    return partial(torch.matmul, matvec)
