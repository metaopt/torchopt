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
from torchopt.linear_solve.utils import make_ridge_matvec, materialize_matvec
from torchopt.typing import TensorTree


__all__ = ['solve_inv']


def _solve_inv(
    matvec: Callable[[TensorTree], TensorTree],  # (x) -> A @ x
    b: TensorTree,
    ridge: Optional[float] = None,
    ns: bool = False,
    **kwargs,
) -> TensorTree:
    """Solves ``A x = b`` using matrix inversion.

    If ``ns = False``, this assumes the matrix ``A`` is a constant matrix and will materialize it
    in memory.

    Args:
        matvec: A function that returns the product between ``A`` and a vector.
        b: A tensor for the right hand side of the equation.
        ridge: Optional ridge regularization. Solves the equation for ``(A + ridge * I) @ x = b``.
        ns: Whether to use Neumann Series matrix inversion approximation. If :data:`False`,
            materialize the matrix ``A`` in memory and use :func:`torch.linalg.solve` instead.
        **kwargs: Additional keyword arguments for the Neumann Series matrix inversion approximation
            solver :func:`torchopt.linalg.ns`.

    Returns:
        The solution with the same shape as ``b``.
    """
    if ridge is not None:
        #      (x) -> A @ x + ridge * x
        # i.e. (x) -> (A + ridge * I) @ x
        matvec = make_ridge_matvec(matvec, ridge=ridge)

    b_flat = pytree.tree_leaves(b)
    if len(b_flat) == 1 and b_flat[0].ndim == 0:
        A, *_ = materialize_matvec(matvec, b)
        return pytree.tree_truediv(b, A)

    if ns:
        return linalg.ns(matvec, b, **kwargs)

    A, _, tree_ravel, tree_unravel = materialize_matvec(matvec, b)
    return tree_unravel(pytree.tree_map(torch.linalg.solve, A, tree_ravel(b)))


def solve_inv(**kwargs):
    """A wrapper that returns a solver function to solve ``A x = b`` using matrix inversion.

    If ``ns = False``, this assumes the matrix ``A`` is a constant matrix and will materialize it
    in memory.

    Args:
        ridge: Optional ridge regularization. Solves the equation for ``(A + ridge * I) @ x = b``.
        ns: Whether to use Neumann Series matrix inversion approximation. If :data:`False`,
            materialize the matrix ``A`` in memory and use :func:`torch.linalg.solve` instead.
        **kwargs: Additional keyword arguments for the Neumann Series matrix inversion approximation
            solver :func:`torchopt.linalg.ns`.

    Returns:
        A solver function with signature ``(matvec, b) -> x`` that solves ``A x = b`` using matrix
        inversion where ``matvec(v) = A v``.

    See Also:
        Neumann Series matrix inversion approximation :func:`torchopt.linalg.ns`.

    Example::

        >>> A = {'a': torch.eye(5, 5), 'b': torch.eye(3, 3)}
        >>> x = {'a': torch.randn(5), 'b': torch.randn(3)}
        >>> def matvec(x: TensorTree) -> TensorTree:
        ...     return {'a': A['a'] @ x['a'], 'b': A['b'] @ x['b']}
        >>> b = matvec(x)
        >>> solver = solve_inv(ns=True, maxiter=10)
        >>> x_hat = solver(matvec, b)
        >>> assert torch.allclose(x_hat['a'], x['a']) and torch.allclose(x_hat['b'], x['b'])

    """
    return functools.partial(_solve_inv, **kwargs)
