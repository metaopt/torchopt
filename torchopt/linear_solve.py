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
# https://github.com/google/jaxopt/blob/main/jaxopt/_src/implicit_diff.py
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
"""Linear algebra solvers."""

# pylint: disable=invalid-name

import functools
from typing import Callable, Optional

import functorch

from torchopt import linalg, pytree
from torchopt.typing import TensorTree


__all__ = ['solve_cg', 'solve_normal_cg']


def tree_add(tree_x: TensorTree, tree_y: TensorTree, alpha: float = 1.0) -> TensorTree:
    """Computes tree_x + alpha * tree_y."""
    return pytree.tree_map(lambda x, y: x.add(y, alpha=alpha), tree_x, tree_y)


def _make_rmatvec(
    matvec: Callable[[TensorTree], TensorTree], x: TensorTree
) -> Callable[[TensorTree], TensorTree]:
    """Returns a function that computes A^T y from matvec(x) = A x."""
    _, vjp, *_ = functorch.vjp(matvec, x)
    return lambda y: vjp(y)[0]


def _normal_matvec(matvec: Callable[[TensorTree], TensorTree], x: TensorTree) -> TensorTree:
    """Computes A^T A x from matvec(x) = A x."""
    matvec_x, vjp, *_ = functorch.vjp(matvec, x)
    return vjp(matvec_x)[0]


def _make_ridge_matvec(
    matvec: Callable[[TensorTree], TensorTree], ridge: float = 0.0
) -> Callable[[TensorTree], TensorTree]:
    def ridge_matvec(v: TensorTree) -> TensorTree:
        return tree_add(matvec(v), v, alpha=ridge)

    return ridge_matvec


def solve_cg(
    matvec: Callable[['TensorTree'], 'TensorTree'],  # (x) -> A @ x
    b: 'TensorTree',
    ridge: Optional[float] = None,
    init: Optional['TensorTree'] = None,
    **kwargs,
) -> 'TensorTree':
    """Solves ``A x = b`` using conjugate gradient.

    It assumes that ``A`` is a hermitian, positive definite matrix.

    Args:
        matvec: A function that returns the product between ``A`` and a vector.
        b: A tree of tensors for the right hand side of the equation.
        ridge: Optional ridge regularization.
        init: Optional initialization to be used by conjugate gradient.
        **kwargs: Additional keyword arguments for the conjugate gradient solver.

    Returns:
        The solution with the same structure as ``b``.
    """
    if ridge is not None:
        #      (x) -> A @ x + ridge * x
        # i.e. (x) -> (A + ridge * I) @ x
        matvec = _make_ridge_matvec(matvec, ridge=ridge)

    # Returns solution for `(A + ridge * I) @ x = b`.
    return linalg.cg(matvec, b, x0=init, **kwargs)


def _solve_normal_cg(
    matvec: Callable[[TensorTree], TensorTree],  # (x) -> A @ x
    b: TensorTree,
    is_sdp: bool = False,
    ridge: Optional[float] = None,
    init: Optional[TensorTree] = None,
    **kwargs,
) -> TensorTree:
    """Solves the normal equation ``A^T A x = A^T b`` using conjugate gradient.

    This can be used to solve ``A x = b`` using conjugate gradient when ``A`` is not hermitian,
    positive definite.

    Args:
        matvec: A function that returns the product between ``A`` and a vector.
        b: A tree of tensors for the right hand side of the equation.
        is_sdp: Whether to assume matrix ``A`` is symmetric definite positive to speedup computation.
        ridge: Optional ridge regularization. Solves the equation for ``(A.T @ A + ridge * I) @ x = A.T @ b``.
        init: Optional initialization to be used by normal conjugate gradient.
        **kwargs: Additional keyword arguments for the conjugate gradient solver.

    Returns:
        The solution with the same structure as ``b``.
    """
    if init is None:
        example_x = b  # This assumes that matvec is a square linear operator.
    else:
        example_x = init

    if is_sdp:
        if ridge is not None:
            raise ValueError('ridge must be specified with `is_sdp=False`.')
        # Returns solution for `A @ x = b`.
        return linalg.cg(matvec, b, x0=init, **kwargs)

    rmatvec = _make_rmatvec(matvec, example_x)  # (x) -> A.T @ x

    def normal_matvec(x):  # (x) -> A.T @ A @ x
        return _normal_matvec(matvec, x)

    if ridge is not None:
        #      (x) -> A.T @ A @ x + ridge * x
        # i.e. (x) -> (A.T @ A + ridge * I) @ x
        normal_matvec = _make_ridge_matvec(normal_matvec, ridge=ridge)

    Ab = rmatvec(b)  # A.T @ b

    # Returns solution for `(A.T @ A + ridge * I) @ x = A.T @ b`.
    return linalg.cg(normal_matvec, Ab, x0=init, **kwargs)


def solve_normal_cg(**kwargs):
    """Wrapper for :func:`solve_normal_cg`."""
    partial_fn = functools.partial(_solve_normal_cg, **kwargs)
    setattr(partial_fn, 'is_sdp', kwargs.get('is_sdp', False))
    return partial_fn
