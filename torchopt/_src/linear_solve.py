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

from typing import Any
from typing import Callable
from typing import Optional
import jax
import functorch
from torchopt._src import linalg


def tree_add_scalar_mul(tree_x, scalar, tree_y):
    """Compute tree_x + scalar * tree_y."""
    return jax.tree_util.tree_map(lambda x, y: x.add(y, alpha=scalar), tree_x, tree_y)


def _make_ridge_matvec(matvec: Callable, ridge: float = 0.0):
    def ridge_matvec(v: Any) -> Any:
        return tree_add_scalar_mul(matvec(v), ridge, v)
    return ridge_matvec


def solve_cg(matvec: Callable,
             b: Any,
             ridge: Optional[float] = None,
             init: Optional[Any] = None,
             **kwargs) -> Any:
    """Solves ``A x = b`` using conjugate gradient.

    It assumes that ``A`` is  a Hermitian, positive definite matrix.

    Args:
      matvec: product between ``A`` and a vector.
      b: pytree.
      ridge: optional ridge regularization.
      init: optional initialization to be used by conjugate gradient.
      **kwargs: additional keyword arguments for solver.

    Returns:
      pytree with same structure as ``b``.
    """
    if ridge is not None:
        matvec = _make_ridge_matvec(matvec, ridge=ridge)
    return linalg.cg(matvec, b, x0=init, **kwargs)[0]


def _make_rmatvec(matvec, x):
    matvec_x, vjp = functorch.vjp(matvec, x)
    return lambda y: vjp(y)[0]
    # transpose = jax.linear_transpose(matvec, x)
    # return lambda y: transpose(y)[0]


def _normal_matvec(matvec, x):
    """Computes A^T A x from matvec(x) = A x."""
    matvec_x, vjp = functorch.vjp(matvec, x)
    return vjp(matvec_x)[0]


def solve_normal_cg(matvec: Callable,
                    b: Any,
                    ridge: Optional[float] = None,
                    init: Optional[Any] = None,
                    **kwargs) -> Any:
    """Solves the normal equation ``A^T A x = A^T b`` using conjugate gradient.

    This can be used to solve Ax=b using conjugate gradient when A is not
    hermitian, positive definite.

    Args:
      matvec: product between ``A`` and a vector.
      b: pytree.
      ridge: optional ridge regularization.
      init: optional initialization to be used by normal conjugate gradient.
      **kwargs: additional keyword arguments for solver.

    Returns:
      pytree with same structure as ``b``.
    """
    if init is None:
        example_x = b  # This assumes that matvec is a square linear operator.
    else:
        example_x = init

    try:
        rmatvec = _make_rmatvec(matvec, example_x)
    except TypeError:
        raise TypeError("The initialization `init` of solve_normal_cg is "
                        "compulsory when `matvec` is nonsquare. It should "
                        "have the same pytree structure as a solution. "
                        "Typically, a pytree filled with zeros should work.")

    def normal_matvec(x):
        return _normal_matvec(matvec, x)

    if ridge is not None:
        normal_matvec = _make_ridge_matvec(normal_matvec, ridge=ridge)

    Ab = rmatvec(b)  # A.T b

    return linalg.cg(normal_matvec, Ab, x0=init, **kwargs)[0]
