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

from __future__ import annotations

import itertools
from typing import Callable

import torch

from torchopt import pytree
from torchopt.typing import TensorTree


def cat_shapes(tree: TensorTree) -> tuple[int, ...]:
    """Concatenate the shapes of the leaves of a tree of tensors."""
    leaves = pytree.tree_leaves(tree)
    return tuple(itertools.chain.from_iterable(tuple(leaf.shape) for leaf in leaves))


def normalize_matvec(
    matvec: TensorTree | Callable[[TensorTree], TensorTree],
) -> Callable[[TensorTree], TensorTree]:
    """Normalize an argument for computing matrix-vector product."""
    if callable(matvec):
        return matvec

    mat_flat, treespec = pytree.tree_flatten(matvec)
    for mat in mat_flat:
        if not isinstance(mat, torch.Tensor) or mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise TypeError(f'Linear operator must be a square matrix, but has shape: {mat.shape}')

    def _matvec(x: TensorTree) -> TensorTree:
        x_flat = pytree.tree_leaves(x)
        if len(x_flat) != len(mat_flat):
            raise ValueError(
                f'`x` must have the same number of leaves as `matvec`, '
                f'but has {len(x_flat)} leaves and `matvec` has {len(mat_flat)} leaves',
            )

        y_flat = map(torch.matmul, mat_flat, x_flat)
        return pytree.tree_unflatten(treespec, y_flat)

    return _matvec
