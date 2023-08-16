# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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
"""The PyTree utilities."""

from __future__ import annotations

import functools
import operator
from typing import Callable

import optree
import optree.typing as typing  # pylint: disable=unused-import
import torch
import torch.distributed.rpc as rpc
from optree import *  # pylint: disable=wildcard-import,unused-wildcard-import

from torchopt.typing import Future, RRef, Scalar, T, TensorTree


__all__ = [
    *optree.__all__,
    'tree_flatten_as_tuple',
    'tree_pos',
    'tree_neg',
    'tree_add',
    'tree_add_scalar_mul',
    'tree_sub',
    'tree_sub_scalar_mul',
    'tree_mul',
    'tree_matmul',
    'tree_scalar_mul',
    'tree_truediv',
    'tree_vdot_real',
    'tree_wait',
]


def tree_flatten_as_tuple(
    tree: PyTree[T],
    is_leaf: Callable[[T], bool] | None = None,
    *,
    none_is_leaf: bool = False,
    namespace: str = '',
) -> tuple[tuple[T, ...], PyTreeSpec]:
    """Flatten a pytree to a tuple of leaves and a PyTreeSpec.

    Args:
        tree (pytree): The pytree to flatten.
        is_leaf (callable or None, optional): An optionally specified function that returns
            :data:`True` if a given node is a leaf. (default: :data:`None`)
        none_is_leaf (bool, optional): If :data:`True`, :data:`None` is considered a leaf rather
            than a internal node with no children. (default: :data:`False`)
        namespace (str, optional): The namespace of custom tree node types. (default: :const:`''`)

    Returns:
        A tuple of (leaves, treespec).
    """
    leaves, treespec = tree_flatten(tree, is_leaf, none_is_leaf=none_is_leaf, namespace=namespace)
    return tuple(leaves), treespec


def acc_add(*args: T) -> T:
    """Accumulate addition."""
    return functools.reduce(operator.add, args)


def acc_mul(*args: T) -> T:
    """Accumulate multiplication."""
    return functools.reduce(operator.mul, args)


def acc_matmul(*args: T) -> T:
    """Accumulate matrix multiplication."""
    return functools.reduce(operator.matmul, args)


def tree_pos(tree: PyTree[T]) -> PyTree[T]:
    """Apply :func:`operator.pos` over leaves."""
    return tree_map(operator.pos, tree)


def tree_neg(tree: PyTree[T]) -> PyTree[T]:
    """Apply :func:`operator.neg` over leaves."""
    return tree_map(operator.neg, tree)


def tree_add(*trees: PyTree[T]) -> PyTree[T]:
    """Tree addition over leaves."""
    return tree_map(acc_add, *trees)


def tree_add_scalar_mul(
    tree_x: TensorTree,
    tree_y: TensorTree,
    alpha: Scalar | None = None,
) -> TensorTree:
    """Compute ``tree_x + alpha * tree_y``."""
    if alpha is None:
        return tree_map(lambda x, y: x.add(y), tree_x, tree_y)
    return tree_map(lambda x, y: x.add(y, alpha=alpha), tree_x, tree_y)


def tree_sub(minuend_tree: PyTree[T], subtrahend_tree: PyTree[T]) -> PyTree[T]:
    """Tree subtraction over leaves."""
    return tree_map(operator.sub, minuend_tree, subtrahend_tree)


def tree_sub_scalar_mul(
    tree_x: TensorTree,
    tree_y: TensorTree,
    alpha: Scalar | None = None,
) -> TensorTree:
    """Compute ``tree_x - alpha * tree_y``."""
    if alpha is None:
        return tree_map(lambda x, y: x.sub(y), tree_x, tree_y)
    return tree_map(lambda x, y: x.sub(y, alpha=alpha), tree_x, tree_y)


def tree_mul(*trees: PyTree[T]) -> PyTree[T]:
    """Tree multiplication over leaves."""
    return tree_map(acc_mul, *trees)


def tree_matmul(*trees: PyTree[T]) -> PyTree[T]:
    """Tree matrix multiplication over leaves."""
    return tree_map(acc_matmul, *trees)


def tree_scalar_mul(scalar: Scalar, multiplicand_tree: PyTree[T]) -> PyTree[T]:
    """Tree scalar multiplication over leaves."""
    return tree_map(lambda x: scalar * x, multiplicand_tree)


def tree_truediv(dividend_tree: PyTree[T], divisor_tree: PyTree[T]) -> PyTree[T]:
    """Tree division over leaves."""
    return tree_map(operator.truediv, dividend_tree, divisor_tree)


def _vdot_real_kernel(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute ``dot(x.conj(), y).real``."""
    x = x.contiguous().view(-1)
    y = y.contiguous().view(-1)
    vdot = torch.dot(x.real, y.real).item()
    if x.is_complex() and y.is_complex():
        vdot += torch.dot(x.imag, y.imag).item()
    return vdot


def tree_vdot_real(tree_x: TensorTree, tree_y: TensorTree) -> float:
    """Compute ``dot(tree_x.conj(), tree_y).real.sum()``."""
    leaves_x, treespec = tree_flatten(tree_x)
    leaves_y = treespec.flatten_up_to(tree_y)
    return sum(map(_vdot_real_kernel, leaves_x, leaves_y))  # type: ignore[arg-type]


def tree_wait(future_tree: PyTree[Future[T]]) -> PyTree[T]:
    r"""Convert a tree of :class:`Future`\s to a tree of results."""
    futures, treespec = tree_flatten(future_tree)

    results = torch.futures.wait_all(futures)

    return tree_unflatten(treespec, results)


if rpc.is_available():  # pragma: no cover

    def tree_as_rref(tree: PyTree[T]) -> PyTree[RRef[T]]:
        r"""Convert a tree of local objects to a tree of :class:`RRef`\s."""
        # pylint: disable-next=import-outside-toplevel,redefined-outer-name,reimported
        from torch.distributed.rpc import RRef

        return tree_map(RRef, tree)

    def tree_to_here(
        rref_tree: PyTree[RRef[T]],
        timeout: float = rpc.api.UNSET_RPC_TIMEOUT,
    ) -> PyTree[T]:
        r"""Convert a tree of :class:`RRef`\s to a tree of local objects."""
        return tree_map(lambda x: x.to_here(timeout=timeout), rref_tree)

    def tree_local_value(rref_tree: PyTree[RRef[T]]) -> PyTree[T]:
        r"""Return the local value of a tree of :class:`RRef`\s."""
        return tree_map(lambda x: x.local_value(), rref_tree)

    __all__ += ['tree_as_rref', 'tree_to_here']


del optree, rpc
