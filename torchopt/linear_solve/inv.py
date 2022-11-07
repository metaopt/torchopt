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
# https://github.com/google/jaxopt/blob/main/jaxopt/_src/linear_solve.py
# ==============================================================================
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Linear algebra solver for ``A x = b`` using matrix inversion."""

# pylint: disable=invalid-name

import functools
from typing import Callable, Optional

import torch

from torchopt import linalg, pytree
from torchopt.linalg.utils import cat_shapes
from torchopt.linear_solve.utils import make_ridge_matvec, materialize_array
from torchopt.typing import TensorTree


__all__ = ['solve_inv']


def _solve_inv(
    matvec: Callable[[TensorTree], TensorTree],  # (x) -> A @ x
    b: TensorTree,
    ridge: Optional[float] = None,
    ns: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Solves ``A x = b`` using matrix inversion.

    It will materialize the matrix ``A`` in memory.

    Args:
        matvec: A function that returns the product between ``A`` and a vector.
        b: A tensor for the right hand side of the equation.
        ridge: Optional ridge regularization.
        ns: Whether to use Neumann Series approximation. If :data:`False`, use
            :func`torch.linalg.inv` instead.

    Returns:
        The solution with the same shape as ``b``.
    """
    if ridge is not None:
        matvec = make_ridge_matvec(matvec, ridge=ridge)
    leaves = pytree.tree_leaves(b)
    if len(leaves) > 0:
        dtype = leaves[0].dtype
    else:
        dtype = None

    shapes = cat_shapes(b)
    if len(shapes) == 0:
        return b / materialize_array(matvec, shapes, dtype=dtype)
    if len(shapes) == 1:
        if ns:
            return linalg.ns(matvec, b, **kwargs)
        A = materialize_array(matvec, shapes, dtype=dtype)
        return pytree.tree_matmul(torch.linalg.inv(A), b)  # type: ignore[return-value]
    raise NotImplementedError


def solve_inv(**kwargs):
    """Wrapper for :func:`solve_inv`."""
    return functools.partial(_solve_inv, **kwargs)
