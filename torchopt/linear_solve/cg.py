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
"""Linear algebra solver for ``A x = b`` using conjugate gradient."""

# pylint: disable=invalid-name

from __future__ import annotations

import functools
from typing import Any, Callable

from torchopt import linalg
from torchopt.linear_solve.utils import make_ridge_matvec
from torchopt.typing import LinearSolver, TensorTree


__all__ = ['solve_cg']


def _solve_cg(
    matvec: Callable[[TensorTree], TensorTree],  # (x) -> A @ x
    b: TensorTree,
    ridge: float | None = None,
    init: TensorTree | None = None,
    **kwargs: Any,
) -> TensorTree:
    """Solve ``A x = b`` using conjugate gradient.

    This assumes that ``A`` is a hermitian, positive definite matrix.

    Args:
        matvec (callable): A function that returns the product between ``A`` and a vector.
        b (Tensor or tree of Tensor): A tree of tensors for the right hand side of the equation.
        ridge (float or None, optional): Optional ridge regularization. If provided, solves the
            equation for ``A x + ridge x = b``. (default: :data:`None`)
        init (Tensor, tree of Tensor, or None, optional): Optional initialization to be used by
            conjugate gradient. If :data:`None`, uses zero initialization. (default: :data:`None`)
        **kwargs: Additional keyword arguments for the conjugate gradient solver.

    Returns:
        The solution with the same structure as ``b``.
    """
    if ridge is not None:
        #      (x) -> A @ x + ridge * x
        # i.e. (x) -> (A + ridge * I) @ x
        matvec = make_ridge_matvec(matvec, ridge=ridge)

    # Returns solution for `(A + ridge * I) @ x = b`.
    return linalg.cg(matvec, b, x0=init, **kwargs)


def solve_cg(**kwargs: Any) -> LinearSolver:
    """Return a solver function to solve ``A x = b`` using conjugate gradient.

    This assumes that ``A`` is a hermitian, positive definite matrix.

    Args:
        ridge (float or None, optional): Optional ridge regularization. If provided, solves the
            equation for ``A x + ridge x = b``. (default: :data:`None`)
        init (Tensor, tree of Tensor, or None, optional): Optional initialization to be used by
            conjugate gradient. If :data:`None`, uses zero initialization. (default: :data:`None`)
        **kwargs: Additional keyword arguments for the conjugate gradient solver
            :func:`torchopt.linalg.cg`.

    Returns:
        A solver function with signature ``(matvec, b) -> x`` that solves ``A x = b`` using
        conjugate gradient where ``matvec(v) = A v``.

    See Also:
        Conjugate gradient iteration :func:`torchopt.linalg.cg`.

    Examples:
        >>> A = {'a': torch.eye(5, 5), 'b': torch.eye(3, 3)}
        >>> x = {'a': torch.randn(5), 'b': torch.randn(3)}
        >>> def matvec(x: TensorTree) -> TensorTree:
        ...     return {'a': A['a'] @ x['a'], 'b': A['b'] @ x['b']}
        >>> b = matvec(x)
        >>> solver = solve_cg(init={'a': torch.zeros(5), 'b': torch.zeros(3)})
        >>> x_hat = solver(matvec, b)
        >>> assert torch.allclose(x_hat['a'], x['a']) and torch.allclose(x_hat['b'], x['b'])
    """
    return functools.partial(_solve_cg, **kwargs)
