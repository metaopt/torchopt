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

import functools
from typing import Callable, Optional, Union

import torch

from torchopt import pytree
from torchopt.linalg.utils import cat_shapes, normalize_matvec
from torchopt.linear_solve.utils import materialize_matvec
from torchopt.typing import TensorTree


__all__ = ['ns', 'ns_inv']


def _ns_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    maxiter: int,
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """Uses Neumann Series Matrix Inversion Approximation to solve ``Ax = b``."""
    if A.ndim != 2:
        raise ValueError(f'`A` must be a 2D tensor, but has shape: {A.shape}')
    ndim = b.ndim
    if ndim == 0:
        raise ValueError(f'`b` must be a vector, but has shape: {b.shape}')
    if ndim >= 2:
        if any(size != 1 for size in b.shape[1:]):
            raise ValueError(f'`b` must be a vector, but has shape: {b.shape}')
        b = b[(...,) + (0,) * (ndim - 1)]  # squeeze trailing dimensions

    inv_A_hat_b = b
    term = b
    if alpha is not None:
        for _ in range(maxiter):
            term = term - alpha * (A @ term)
            inv_A_hat_b = inv_A_hat_b + term
    else:
        for _ in range(maxiter):
            term = term - A @ term
            inv_A_hat_b = inv_A_hat_b + term

    if ndim >= 2:
        inv_A_hat_b = inv_A_hat_b[(...,) + (None,) * (ndim - 1)]  # unqueeze trailing dimensions
    return inv_A_hat_b


def ns(
    A: Union[Callable[[TensorTree], TensorTree], torch.Tensor],
    b: TensorTree,
    maxiter: Optional[int] = None,
    *,
    alpha: Optional[float] = None,
) -> TensorTree:
    """Uses Neumann Series Matrix Inversion Approximation to solve ``Ax = b``.

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
    if maxiter is None:
        maxiter = 10
    b_flat = pytree.tree_leaves(b)
    if len(b_flat) == 0:
        raise ValueError('`b` must be a non-empty pytree.')
    if len(b_flat) >= 2:
        raise ValueError('`b` must be a pytree with a single leaf.')
    b_leaf = b_flat[0]
    if b_leaf.ndim >= 2 and any(size != 1 for size in b.shape[1:]):
        raise ValueError(f'`b` must be a vector or a scalar, but has shape: {b_leaf.shape}')

    matvec = normalize_matvec(A)
    A: TensorTree = materialize_matvec(matvec, b)
    return pytree.tree_map(functools.partial(_ns_solve, maxiter=maxiter, alpha=alpha), A, b)


def _ns_inv(A: torch.Tensor, maxiter: int, alpha: Optional[float] = None):
    """Uses Neumann Series iteration to solve ``A^{-1}``."""
    if A.ndim != 2:
        raise ValueError(f'`A` must be a 2D tensor, but has shape: {A.shape}')

    I = torch.eye(*A.shape, out=torch.empty_like(A))
    inv_A_hat = torch.zeros_like(A)
    if alpha is not None:
        for rank in range(maxiter):
            inv_A_hat = inv_A_hat + torch.linalg.matrix_power(I - alpha * A, rank)
    else:
        for rank in range(maxiter):
            inv_A_hat = inv_A_hat + torch.linalg.matrix_power(I - A, rank)
    return inv_A_hat


def ns_inv(
    A: TensorTree,
    maxiter: Optional[int] = None,
    *,
    alpha: Optional[float] = None,
) -> TensorTree:
    """Uses Neumann Series iteration to solve ``A^{-1}``.

    Args:
        A: (tensor or tree of tensors or function)
            2D array or function that calculates the linear map (matrix-vector product) ``Ax`` when
            called like ``A(x)``. ``A`` must represent a hermitian, positive definite matrix, and
            must return array(s) with the same structure and shape as its argument.
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

    if isinstance(A, torch.Tensor):
        I = torch.eye(*A.shape, out=torch.empty_like(A))
        inv_A_hat = torch.zeros_like(A)
        if alpha is not None:
            for rank in range(maxiter):
                inv_A_hat = inv_A_hat + torch.linalg.matrix_power(I - alpha * A, rank)
        else:
            for rank in range(maxiter):
                inv_A_hat = inv_A_hat + torch.linalg.matrix_power(I - A, rank)
        return inv_A_hat

    return pytree.tree_map(functools.partial(_ns_inv, maxiter=maxiter, alpha=alpha), A)
