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
"""Linear algebra solver for ``A^T A x = A^T b`` using conjugate gradient."""

# pylint: disable=invalid-name

import functools
from typing import Callable, Optional

from torchopt import linalg
from torchopt.linear_solve.utils import make_normal_matvec, make_ridge_matvec, make_rmatvec
from torchopt.typing import TensorTree


__all__ = ['solve_normal_cg']


def _solve_normal_cg(
    matvec: Callable[[TensorTree], TensorTree],  # (x) -> A @ x
    b: TensorTree,
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
        ridge: Optional ridge regularization. Solves the equation for ``(A.T @ A + ridge * I) @ x = A.T @ b``.
        init: Optional initialization to be used by normal conjugate gradient.
        **kwargs: Additional keyword arguments for the conjugate gradient solver
            :func:`torchopt.linalg.cg`.

    Returns:
        The solution with the same structure as ``b``.
    """
    if init is None:
        example_x = b  # This assumes that matvec is a square linear operator.
    else:
        example_x = init

    rmatvec = make_rmatvec(matvec, example_x)  # (x) -> A.T @ x
    normal_matvec = make_normal_matvec(matvec)  # (x) -> A.T @ A @ x

    if ridge is not None:
        #      (x) -> A.T @ A @ x + ridge * x
        # i.e. (x) -> (A.T @ A + ridge * I) @ x
        normal_matvec = make_ridge_matvec(normal_matvec, ridge=ridge)

    rhs = rmatvec(b)  # A.T @ b

    # Returns solution for `(A.T @ A + ridge * I) @ x = A.T @ b`.
    return linalg.cg(normal_matvec, rhs, x0=init, **kwargs)


def solve_normal_cg(**kwargs):
    """A wrapper that returns a solver function to solve ``A^T A x = A^T b`` using conjugate gradient.

    This can be used to solve ``A x = b`` using conjugate gradient when ``A`` is not hermitian,
    positive definite.

    Args:
        ridge: Optional ridge regularization. Solves the equation for ``(A.T @ A + ridge * I) @ x = A.T @ b``.
        init: Optional initialization to be used by normal conjugate gradient.
        **kwargs: Additional keyword arguments for the conjugate gradient solver
            :func:`torchopt.linalg.cg`.

    Returns:
        A solver function with signature ``(matvec, b) -> x`` that solves ``A^T A x = A^T b`` using
        conjugate gradient where ``matvec(v) = A v``.

    See Also:
        Conjugate gradient iteration :func:`torchopt.linalg.cg`.

    Example::

        >>> A = {'a': torch.randn(5, 5), 'b': torch.randn(3, 3)}
        >>> x = {'a': torch.randn(5), 'b': torch.randn(3)}
        >>> def matvec(x: TensorTree) -> TensorTree:
        ...     return {'a': A['a'] @ x['a'], 'b': A['b'] @ x['b']}
        >>> b = matvec(x)
        >>> solver = solve_normal_cg(init={'a': torch.zeros(5), 'b': torch.zeros(3)})
        >>> x_hat = solver(matvec, b)
        >>> assert torch.allclose(x_hat['a'], x['a']) and torch.allclose(x_hat['b'], x['b'])

    """
    return functools.partial(_solve_normal_cg, **kwargs)
