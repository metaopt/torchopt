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
import math
from functools import partial

import jax
import torch


# aliases for working with pytrees
def _vdot_real_part(x, y):
    x = x.view(-1)
    y = y.view(-1)
    result = torch.dot(x.real, y.real)
    if x.is_complex() or y.is_complex():
        result += torch.dot(x.imag, y.imag)
    return result


def _vdot_real_tree(x, y):
    return jax.tree_util.tree_map(_vdot_real_part, x, y)


def _identity(x):
    return x


def _normalize_matvec(f):
    """Normalize an argument for computing matrix-vector products."""
    if callable(f):
        return f
    elif isinstance(f, torch.Tensor):
        if f.ndim != 2 or f.shape[0] != f.shape[1]:
            raise ValueError(f'linear operator must be a square matrix, but has shape: {f.shape}')
        return partial(torch.matmul, f)
    else:
        # TODO(shoyer): handle sparse arrays?
        raise TypeError(f'linear operator must be either a function or ndarray: {f}')


def _safe_sum(obj):
    if isinstance(obj, tuple):
        return sum(obj)
    return obj


def _cg_solve(A, b, x0=None, *, maxiter, tol=1e-5, atol=0.0, M=_identity):
    # tolerance handling uses the "non-legacy" behavior of scipy.sparse.linalg.cg
    bs = _safe_sum(_vdot_real_tree(b, b))
    atol2 = max(tol**2 * bs, atol**2)

    # https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method

    min_rs = math.inf

    def cond_fun(value, min_rs):
        _, r, gamma, _, k = value
        rs = gamma if M is _identity else _safe_sum(_vdot_real_tree(r, r))
        return (rs > atol2) & (k < maxiter) & (rs <= min_rs), rs

    def body_fun(value):
        x, r, gamma, p, k = value
        Ap = A(p)
        alpha = gamma / _safe_sum(_vdot_real_tree(p, Ap))
        x_ = jax.tree_util.tree_map(lambda a, b: a.add(b, alpha=alpha), x, p)
        r_ = jax.tree_util.tree_map(lambda a, b: a.sub(b, alpha=alpha), r, Ap)
        z_ = M(r_)
        gamma_ = _safe_sum(_vdot_real_tree(r_, z_))
        beta_ = gamma_ / gamma
        p_ = jax.tree_util.tree_map(lambda a, b: a.add(b, alpha=beta_), z_, p)
        return x_, r_, gamma_, p_, k + 1

    r0 = jax.tree_util.tree_map(torch.sub, b, A(x0))
    p0 = z0 = M(r0)
    gamma0 = _safe_sum(_vdot_real_tree(r0, z0))
    initial_value = (x0, r0, gamma0, p0, 0)

    value = initial_value
    not_stop, min_rs = cond_fun(value, min_rs)
    while not_stop:
        value = body_fun(value)
        not_stop, rs = cond_fun(value, min_rs)
        min_rs = min(rs, min_rs)

    x_final, *_ = value

    return x_final


def _shapes(pytree):
    flatten_tree, _ = jax.tree_util.tree_flatten(pytree)
    return jax.tree_util.tree_flatten([tuple(term.shape) for term in flatten_tree])[0]


def _isolve(
    _isolve_solve, A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None, check_symmetric=False
):
    if x0 is None:
        x0 = jax.tree_util.tree_map(torch.zeros_like, b)

    if maxiter is None:
        size = sum(_shapes(b))
        maxiter = size  # copied from scipy

    if M is None:
        M = _identity
    A = _normalize_matvec(A)
    M = _normalize_matvec(M)

    if _shapes(x0) != _shapes(b):
        raise ValueError(
            'arrays in x0 and b must have matching shapes: ' f'{_shapes(x0)} vs {_shapes(b)}'
        )

    isolve_solve = partial(_isolve_solve, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M)

    x = isolve_solve(A, b)
    info = None
    return x, info


def cg(A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None):
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    The numerics of JAX's ``cg`` should exact match SciPy's ``cg`` (up to
    numerical precision), but note that the interface is slightly different: you
    need to supply the linear operator ``A`` as a function instead of a sparse
    matrix or ``LinearOperator``.

    Derivatives of ``cg`` are implemented via implicit differentiation with
    another ``cg`` solve, rather than by differentiating *through* the solver.
    They will be accurate only if both solves converge.

    Args:
        A: ndarray or function
            2D array or function that calculates the linear map (matrix-vector
            product) ``Ax`` when called like ``A(x)``. ``A`` must represent a
            hermitian, positive definite matrix, and must return array(s) with the
            same structure and shape as its argument.
        b: array or tree of arrays
            Right hand side of the linear system representing a single vector. Can be
            stored as an array or Python container of array(s) with any shape.
            x: array or tree of arrays
            The converged solution. Has the same structure as ``b``.
        info: None
            Placeholder for convergence information. In the future, JAX will report
            the number of iterations when convergence is not achieved, like SciPy.
        x0: array
            Starting guess for the solution. Must have the same structure as ``b``.
        tol: float, optional
            Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
            We do not implement SciPy's "legacy" behavior, so JAX's tolerance will
            differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``cg``.
        atol: float, optional
            Tolerances for convergence, ``norm(residual) <= max(tol*norm(b), atol)``.
            We do not implement SciPy's "legacy" behavior, so JAX's tolerance will
            differ from SciPy unless you explicitly pass ``atol`` to SciPy's ``cg``.
        maxiter: integer
            Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        M: ndarray or function
            Preconditioner for A.  The preconditioner should approximate the
            inverse of A.  Effective preconditioning dramatically improves the
            rate of convergence, which implies that fewer iterations are needed
            to reach a given error tolerance.

    Returns:
        the CG linear solver
    """
    return _isolve(
        _cg_solve, A=A, b=b, x0=x0, tol=tol, atol=atol, maxiter=maxiter, M=M, check_symmetric=True
    )
