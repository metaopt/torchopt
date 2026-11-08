# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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

from __future__ import annotations

import torch

import torchopt


def test_normalize_matvec() -> None:
    A = [torch.rand(10, 10) for _ in range(10)]
    x = [torch.rand(10, 1) for _ in range(10)]
    AxFn = torchopt.linalg.utils.normalize_matvec(A)
    Ax = AxFn(x)
    for Ax_item, A_item, x_item in zip(Ax, A, x):
        assert torch.equal(Ax_item, A_item @ x_item)


def test_cg_scalar() -> None:
    def A(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x

    b = torch.tensor(6.0, dtype=torch.float64)

    x = torchopt.linalg.cg(A, b)
    assert torch.allclose(x, torch.tensor(3.0, dtype=torch.float64))

    x_zero = torchopt.linalg.cg(A, b, maxiter=0)
    assert torch.allclose(x_zero, torch.tensor(0.0, dtype=torch.float64))

    x0 = torch.tensor(1.0, dtype=torch.float64)
    x_zero_x0 = torchopt.linalg.cg(A, b, x0=x0, maxiter=0)
    assert torch.allclose(x_zero_x0, x0)


def test_cg_mixed_scalar_pytree() -> None:
    def A(tree: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            'a': 2.0 * tree['a'],
            'b': 3.0 * tree['b'],
        }

    b = {
        'a': torch.tensor(6.0, dtype=torch.float64),
        'b': torch.tensor([9.0, 12.0], dtype=torch.float64),
    }

    sol = torchopt.linalg.cg(A, b)
    assert torch.allclose(sol['a'], torch.tensor(3.0, dtype=torch.float64))
    assert torch.allclose(sol['b'], torch.tensor([3.0, 4.0], dtype=torch.float64))


def test_custom_root_scalar_gradient() -> None:
    def optimality_fn(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return 2.0 * x - theta

    @torchopt.diff.implicit.custom_root(optimality_fn, argnums=1)
    def solver(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return theta / 2.0

    theta = torch.tensor(6.0, dtype=torch.float64, requires_grad=True)
    init_x = torch.tensor(0.0, dtype=torch.float64)
    x = solver(init_x, theta)
    assert torch.allclose(x, torch.tensor(3.0, dtype=torch.float64))

    x.backward()
    assert theta.grad is not None
    assert torch.allclose(theta.grad, torch.tensor(0.5, dtype=torch.float64))
