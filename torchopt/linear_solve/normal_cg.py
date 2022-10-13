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
    """Wrapper for :func:`solve_normal_cg`."""
    partial_fn = functools.partial(_solve_normal_cg, **kwargs)
    setattr(partial_fn, 'is_sdp', kwargs.get('is_sdp', False))
    return partial_fn
