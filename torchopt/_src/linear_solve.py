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

# pylint: disable=invalid-name

import functools
from typing import Callable, Optional

import functorch

from torchopt._src import linalg
from torchopt._src.typing import TensorTree
from torchopt._src.utils import pytree


def tree_add(tree_x: TensorTree, tree_y: TensorTree, alpha: float = 1.0) -> TensorTree:
    """Computes tree_x + alpha * tree_y."""
    return pytree.tree_map(lambda x, y: x.add(y, alpha=alpha), tree_x, tree_y)


def _make_ridge_matvec(
    matvec: Callable[[TensorTree], TensorTree], ridge: float = 0.0
) -> Callable[[TensorTree], TensorTree]:
    def ridge_matvec(v: TensorTree) -> TensorTree:
        return tree_add(matvec(v), v, alpha=ridge)

    return ridge_matvec


def solve_cg(
    matvec: Callable[[TensorTree], TensorTree],
    b: TensorTree,
    ridge: Optional[float] = None,
    init: Optional[TensorTree] = None,
    **kwargs,
) -> TensorTree:
    """Solves ``A x = b`` using conjugate gradient.

    It assumes that ``A`` is a Hermitian, positive definite matrix.

    Args:
        matvec: a function that returns the product between ``A`` and a vector.
        b: a tree of tensors.
        ridge: optional ridge regularization.
        init: optional initialization to be used by conjugate gradient.
        **kwargs: additional keyword arguments for solver.

    Returns:
        The solution with the same structure as ``b``.
    """
    if ridge is not None:
        matvec = _make_ridge_matvec(matvec, ridge=ridge)
    return linalg.cg(matvec, b, x0=init, **kwargs)


def _make_rmatvec(
    matvec: Callable[[TensorTree], TensorTree], x: TensorTree
) -> Callable[[TensorTree], TensorTree]:
    _, vjp, *_ = functorch.vjp(matvec, x)
    return lambda y: vjp(y)[0]


def _normal_matvec(matvec: Callable[[TensorTree], TensorTree], x: TensorTree) -> TensorTree:
    """Computes A^T A x from matvec(x) = A x."""
    matvec_x, vjp, *_ = functorch.vjp(matvec, x)
    return vjp(matvec_x)[0]


def _solve_normal_cg(
    matvec: Callable[[TensorTree], TensorTree],
    b: TensorTree,
    ridge: Optional[float] = None,
    init: Optional[TensorTree] = None,
    **kwargs,
) -> TensorTree:
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
        The solution with the same structure as ``b``.
    """
    if init is None:
        example_x = b  # This assumes that matvec is a square linear operator.
    else:
        example_x = init

    rmatvec = _make_rmatvec(matvec, example_x)

    def normal_matvec(x):
        return _normal_matvec(matvec, x)

    if ridge is not None:
        normal_matvec = _make_ridge_matvec(normal_matvec, ridge=ridge)

    Ab = rmatvec(b)  # A.T b

    return linalg.cg(normal_matvec, Ab, x0=init, **kwargs)


def solve_normal_cg(**kwargs):
    """Wrapper for `solve_normal_cg`."""
    return functools.partial(_solve_normal_cg, **kwargs)
