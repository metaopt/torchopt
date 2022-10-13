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
"""Neumann Series Matrix Inversion Approximation to solve ``Ax = b``."""

# pylint: disable=invalid-name

from typing import Callable, Optional, Union

import torch

from torchopt import pytree
from torchopt.linalg.utils import cat_shapes, normalize_matvec
from torchopt.typing import TensorTree


__all__ = ['ns', 'ns_inv']


def ns(
    A: Union[torch.Tensor, Callable[[TensorTree], TensorTree]],
    b: TensorTree,
    maxiter: Optional[int] = None,
    *,
    alpha: Optional[float] = None,
) -> TensorTree:
    """Use Neumann Series Matrix Inversion Approximation to solve ``Ax = b``.

    Args:
        A: (tensor or tree of tensors or function)
            2D array or function that calculates the linear map (matrix-vector product) ``Ax`` when
            called like ``A(x)``. ``A`` must represent a hermitian, positive definite matrix, and
            must return array(s) with the same structure and shape as its argument.
        b: (tensor or tree of tensors)
            Right hand side of the linear system representing a single vector. Can be stored as an
            array or Python container of array(s) with any shape.
        maxiter: (integer, optional)
            Maximum number of iterations. Iteration will stop after maxiter steps even if the
            specified tolerance has not been achieved.
        alpha: (float, optional)
            Decay coefficient.

    Returns:
        The Neumann Series (NS) matrix inversion approximation.
    """
    A = normalize_matvec(A)
    if maxiter is None:
        size = sum(cat_shapes(b))
        maxiter = 10 * size

    inv_A_hat_b = b
    for _ in range(maxiter):
        if alpha is not None:
            b = pytree.tree_sub(b, pytree.tree_mul(alpha, A(b)))
        else:
            b = pytree.tree_sub(b, A(b))
        inv_A_hat_b = pytree.tree_sub(inv_A_hat_b, b)
    return inv_A_hat_b


def ns_inv(
    A: TensorTree,
    n: int,
    maxiter: Optional[int] = None,
    *,
    alpha: Optional[float] = None,
) -> TensorTree:
    """Use Neumann Series iteration to solve ``A^{-1}``.

    Args:
        A: (tensor or tree of tensors or function)
            2D array or function that calculates the linear map (matrix-vector product) ``Ax`` when
            called like ``A(x)``. ``A`` must represent a hermitian, positive definite matrix, and
            must return array(s) with the same structure and shape as its argument.
        n: (integer)
            Number of rows (and columns) in n x n output.
        maxiter: (integer, optional)
            Maximum number of iterations. Iteration will stop after maxiter steps even if the
            specified tolerance has not been achieved.
        alpha: (float, optional)
            Decay coefficient.

    Returns:
        The Neumann Series (NS) matrix inversion approximation.
    """
    if maxiter is None:
        size = sum(cat_shapes(A))
        maxiter = 10 * size

    I = torch.eye(n)
    inv_A_hat = pytree.tree_map(torch.zeros_like, A)
    if alpha is not None:
        for rank in range(maxiter):
            inv_A_hat += torch.linalg.matrix_power(I - pytree.tree_mul(alpha, A), rank)
    else:
        for rank in range(maxiter):
            inv_A_hat += torch.linalg.matrix_power(I - A, rank)
    return inv_A_hat
