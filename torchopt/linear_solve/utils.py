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
"""Utilities for linear algebra solvers."""

# pylint: disable=invalid-name

from typing import Callable

import functorch

from torchopt import pytree
from torchopt.typing import TensorTree


def tree_add(tree_x: TensorTree, tree_y: TensorTree, alpha: float = 1.0) -> TensorTree:
    """Computes tree_x + alpha * tree_y."""
    return pytree.tree_map(lambda x, y: x.add(y, alpha=alpha), tree_x, tree_y)


def make_rmatvec(
    matvec: Callable[[TensorTree], TensorTree], example_x: TensorTree
) -> Callable[[TensorTree], TensorTree]:
    """Returns a function that computes ``rmatvec(y) = A.T @ y`` from ``matvec(x) = A @ x``."""
    _, vjp, *_ = functorch.vjp(matvec, example_x)

    return lambda y: vjp(y)[0]


def make_normal_matvec(
    matvec: Callable[[TensorTree], TensorTree]
) -> Callable[[TensorTree], TensorTree]:
    """Returns a function that computes ``normal_matvec(y) = A.T @ A @ y`` from ``matvec(x) = A @ x``."""

    def normal_matvec(y: TensorTree) -> TensorTree:
        """Computes ``A.T @ A @ y`` from ``matvec(x) = A @ x``."""
        matvec_y, vjp, *_ = functorch.vjp(matvec, y)
        return vjp(matvec_y)[0]

    return normal_matvec


def make_ridge_matvec(
    matvec: Callable[[TensorTree], TensorTree], ridge: float = 0.0
) -> Callable[[TensorTree], TensorTree]:
    """Returns a function that computes ``ridge_matvec(y) = A.T @ A @ y + ridge * y`` from ``matvec(x) = A @ x``."""

    def ridge_matvec(y: TensorTree) -> TensorTree:
        """Computes ``A.T @ A @ v + ridge * v`` from ``matvec(x) = A @ x``."""
        return tree_add(matvec(y), y, alpha=ridge)

    return ridge_matvec
