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

from __future__ import annotations

import functools
from typing import Callable

import torch

from torchopt import pytree
from torchopt.linalg.utils import normalize_matvec
from torchopt.typing import TensorTree


__all__ = ['ns', 'ns_inv']


def _ns_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    maxiter: int,
    alpha: float | None = None,
) -> torch.Tensor:
    """Use Neumann Series Matrix Inversion Approximation to solve ``Ax = b``."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f'`A` must be a square matrix, but has shape: {A.shape}')

    inv_A_hat_b = b
    v = b
    if alpha is not None:
        # A^{-1} = a [I - (I - a A)]^{-1} = a [I + (I - a A) + (I - a A)^2 + (I - a A)^3 + ...]
        for _ in range(maxiter):
            v = v - alpha * (A @ v)
            inv_A_hat_b = inv_A_hat_b + v
        inv_A_hat_b = alpha * inv_A_hat_b
    else:
        # A^{-1} = [I - (I - A)]^{-1} = I + (I - A) + (I - A)^2 + (I - A)^3 + ...
        for _ in range(maxiter):
            v = v - A @ v
            inv_A_hat_b = inv_A_hat_b + v

    return inv_A_hat_b


def ns(
    A: TensorTree | Callable[[TensorTree], TensorTree],
    b: TensorTree,
    maxiter: int | None = None,
    *,
    alpha: float | None = None,
) -> TensorTree:
    """Use Neumann Series Matrix Inversion Approximation to solve ``Ax = b``.

    Args:
        A (Tensor or tree of Tensor): 2D array or function that calculates the linear map
            (matrix-vector product) ``Ax`` when called like ``A(x)``. ``A`` must represent a
            hermitian, positive definite matrix, and must return tensor(s) with the same structure
            and shape as its argument.
        b (Tensor or tree of Tensor): Right hand side of the linear system representing a single
            vector. Can be stored as a tensor or Python container of tensor(s) with any shape.
        maxiter (int or None, optional): Maximum number of iterations. Iteration will stop after
            maxiter steps even if the specified tolerance has not been achieved. If :data:`None`,
            :const:`10` will be used. (default: :const:`10`)
        alpha: (float or None, optional): Decay coefficient. If :data:`None`, :const:`1.0` will be
            used. (default: :const:`1.0`)

    Returns:
        The Neumann Series (NS) matrix inversion approximation.
    """
    if maxiter is None:
        maxiter = 10

    if not callable(A):
        return pytree.tree_map(functools.partial(_ns_solve, maxiter=maxiter, alpha=alpha), A, b)

    matvec = normalize_matvec(A)
    inv_A_hat_b = b
    v = b
    if alpha is not None:
        # A^{-1} = a [I - (I - a A)]^{-1} = a [I + (I - a A) + (I - a A)^2 + (I - a A)^3 + ...]
        for _ in range(maxiter):
            # v = v - alpha * (A @ v)
            v = pytree.tree_sub_scalar_mul(v, matvec(v), alpha=alpha)
            # inv_A_hat_b = inv_A_hat_b + v
            inv_A_hat_b = pytree.tree_add(inv_A_hat_b, v)
        # inv_A_hat_b = alpha * inv_A_hat_b
        inv_A_hat_b = pytree.tree_scalar_mul(alpha, inv_A_hat_b)
    else:
        # A^{-1} = [I - (I - A)]^{-1} = I + (I - A) + (I - A)^2 + (I - A)^3 + ...
        for _ in range(maxiter):
            # v = v - A @ v
            v = pytree.tree_sub(v, matvec(v))
            # inv_A_hat_b = inv_A_hat_b + v
            inv_A_hat_b = pytree.tree_add(inv_A_hat_b, v)

    return inv_A_hat_b


def _ns_inv(A: torch.Tensor, maxiter: int, alpha: float | None = None) -> torch.Tensor:
    """Use Neumann Series iteration to solve ``A^{-1}``."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f'`A` must be a square matrix, but has shape: {A.shape}')

    I = torch.eye(*A.shape, out=torch.empty_like(A))  # noqa: E741
    inv_A_hat = torch.zeros_like(A)
    if alpha is not None:
        # A^{-1} = a [I - (I - a A)]^{-1} = a [I + (I - a A) + (I - a A)^2 + (I - a A)^3 + ...]
        M = I - alpha * A
        for rank in range(maxiter):
            # pylint: disable-next=not-callable
            inv_A_hat = inv_A_hat + torch.linalg.matrix_power(M, rank)
        inv_A_hat = alpha * inv_A_hat
    else:
        # A^{-1} = [I - (I - A)]^{-1} = I + (I - A) + (I - A)^2 + (I - A)^3 + ...
        M = I - A
        for rank in range(maxiter):
            # pylint: disable-next=not-callable
            inv_A_hat = inv_A_hat + torch.linalg.matrix_power(M, rank)
    return inv_A_hat


def ns_inv(
    A: TensorTree,
    maxiter: int | None = None,
    *,
    alpha: float | None = None,
) -> TensorTree:
    """Use Neumann Series iteration to solve ``A^{-1}``.

    Args:
        A (Tensor or tree of Tensor): 2D array or function that calculates the linear map
            (matrix-vector product) ``Ax`` when called like ``A(x)``. ``A`` must represent a
            hermitian, positive definite matrix, and must return tensor(s) with the same structure
            and shape as its argument.
        maxiter (int or None, optional): Maximum number of iterations. Iteration will stop after
            maxiter steps even if the specified tolerance has not been achieved. If :data:`None`,
            :const:`10` will be used. (default: :const:`10`)
        alpha: (float or None, optional): Decay coefficient. If :data:`None`, :const:`1.0` will be
            used. (default: :const:`1.0`)

    Returns:
        The Neumann Series (NS) matrix inversion approximation.
    """
    if maxiter is None:
        maxiter = 10

    return pytree.tree_map(functools.partial(_ns_inv, maxiter=maxiter, alpha=alpha), A)
