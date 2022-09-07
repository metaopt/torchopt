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
# https://github.com/google/jax/blob/main/jax/_src/scipy/sparse/linalg.py
# ==============================================================================
# Copyright 2020 Google LLC
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

from functools import partial
from typing import Callable, List, Optional, Union

import torch

from torchopt._src.typing import TensorTree
from torchopt._src.utils import pytree


def _inner_product_kernel(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.view(-1)
    y = y.view(-1)
    prod = torch.dot(x.real, y.real).item()
    if x.is_complex() or y.is_complex():
        prod += torch.dot(x.imag, y.imag).item()
    return prod


def _tree_inner_product(
    tree_left: TensorTree,
    tree_right: TensorTree,
    kernel: Callable[[torch.Tensor, torch.Tensor], float] = _inner_product_kernel,
) -> TensorTree:
    leaves_left, treedef = pytree.tree_flatten(tree_left)
    leaves_right = treedef.flatten_up_to(tree_right)
    return sum(map(kernel, leaves_left, leaves_right))


def _identity(x: TensorTree) -> TensorTree:
    return x


def _normalize_matvec(
    f: Union[Callable[[TensorTree], TensorTree], torch.Tensor]
) -> Callable[[TensorTree], TensorTree]:
    """Normalize an argument for computing matrix-vector products."""
    if callable(f):
        return f

    assert isinstance(f, torch.Tensor)
    if f.ndim != 2 or f.shape[0] != f.shape[1]:
        raise ValueError(f'linear operator must be a square matrix, but has shape: {f.shape}')
    return partial(torch.matmul, f)


# pylint: disable-next=too-many-locals
def _cg_solve(
    A: Callable[[TensorTree], TensorTree],
    b: TensorTree,
    x0: TensorTree,
    *,
    maxiter: int,
    rtol: float = 1e-5,
    atol: float = 0.0,
    M: Callable[[TensorTree], TensorTree] = _identity,
) -> TensorTree:
    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    # tolerance handling uses the "non-legacy" behavior of `scipy.sparse.linalg.cg`
    bs = _tree_inner_product(b, b)
    atol2 = max(rtol**2 * bs, atol**2)

    def cond_fun(value):
        _, r, gamma, _, k = value
        rs = gamma if M is _identity else _tree_inner_product(r, r)
        return rs > atol2 and k < maxiter

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / _tree_inner_product(p, Ap)
        x_ = pytree.tree_map(lambda a, b: a.add(b, alpha=alpha), x, p)
        r_ = pytree.tree_map(lambda a, b: a.sub(b, alpha=alpha), r, Ap)
        z_ = M(r_)
        gamma_ = _tree_inner_product(r_, z_)
        beta_ = gamma_ / gamma
        p_ = pytree.tree_map(lambda a, b: a.add(b, alpha=beta_), z_, p)
        return x_, r_, gamma_, p_, k + 1

    r0 = pytree.tree_map(torch.sub, b, A(x0))
    p0 = z0 = M(r0)
    gamma0 = _tree_inner_product(r0, z0)

    value = (x0, r0, gamma0, p0, 0)
    not_stop = cond_fun(value)
    while not_stop:
        value = body_fun(value)
        not_stop = cond_fun(value)

    x_final, *_ = value

    return x_final


def _shapes(tree: TensorTree) -> List[int]:
    flattened = pytree.tree_leaves(tree)
    return pytree.tree_leaves([tuple(term.shape) for term in flattened])


def _isolve(
    _isolve_solve: Callable,
    A: Union[torch.Tensor, Callable[[TensorTree], TensorTree]],
    b: TensorTree,
    x0: Optional[TensorTree] = None,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    M: Optional[Union[torch.Tensor, Callable[[TensorTree], TensorTree]]] = None,
) -> TensorTree:
    if x0 is None:
        x0 = pytree.tree_map(torch.zeros_like, b)

    if maxiter is None:
        size = sum(_shapes(b))
        maxiter = 10 * size  # copied from SciPy

    if M is None:
        M = _identity
    A = _normalize_matvec(A)
    M = _normalize_matvec(M)

    if _shapes(x0) != _shapes(b):
        raise ValueError(
            'arrays in x0 and b must have matching shapes: ' f'{_shapes(x0)} vs {_shapes(b)}'
        )

    isolve_solve = partial(_isolve_solve, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter, M=M)

    x = isolve_solve(A, b)
    return x


def cg(
    A: Union[torch.Tensor, Callable[[TensorTree], TensorTree]],
    b: TensorTree,
    x0: Optional[TensorTree] = None,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    M: Optional[Union[torch.Tensor, Callable[[TensorTree], TensorTree]]] = None,
) -> TensorTree:
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    The numerics of JAX's ``cg`` should exact match SciPy's ``cg`` (up to numerical precision), but
    note that the interface is slightly different: you need to supply the linear operator ``A`` as a
    function instead of a sparse matrix or ``LinearOperator``.

    Derivatives of :func:`cg` are implemented via implicit differentiation with another :func:`cg`
    solve, rather than by differentiating *through* the solver. They will be accurate only if both
    solves converge.

    Args:
        A: (tensor or tree of tensors or function)
            2D array or function that calculates the linear map (matrix-vector
            product) ``Ax`` when called like ``A(x)``. ``A`` must represent a
            hermitian, positive definite matrix, and must return array(s) with the
            same structure and shape as its argument.
        b: (tensor or tree of tensors)
            Right hand side of the linear system representing a single vector. Can be
            stored as an array or Python container of array(s) with any shape.
        x0: (tensor or tree of tensors, optional)
            Starting guess for the solution. Must have the same structure as ``b``.
        rtol: (float, optional, default: :const:`1e-5`)
            Tolerances for convergence, ``norm(residual) <= max(rtol*norm(b), atol)``. We do not
            implement SciPy's "legacy" behavior, so JAX's tolerance will differ from SciPy unless
            you explicitly pass ``atol`` to SciPy's ``cg``.
        atol: (float, optional, default: :const:`0.0`)
            Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``. We do not
            implement SciPy's "legacy" behavior, so JAX's tolerance will differ from SciPy unless
            you explicitly pass ``atol`` to SciPy's ``cg``.
        maxiter: (integer, optional)
            Maximum number of iterations. Iteration will stop after maxiter steps even if the
            specified tolerance has not been achieved.
        M: (tensor or tree of tensors or function)
            Pre-conditioner for ``A``. The pre-conditioner should approximate the inverse of ``A``.
            Effective preconditioning dramatically improves the rate of convergence, which implies
            that fewer iterations are needed to reach a given error tolerance.

    Returns:
        the Conjugate Gradient (CG) linear solver
    """
    return _isolve(_cg_solve, A=A, b=b, x0=x0, rtol=rtol, atol=atol, maxiter=maxiter, M=M)
