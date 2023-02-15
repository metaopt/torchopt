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

import torch

import torchopt


def test_normalize_matvec() -> None:
    A = [torch.rand(10, 10) for _ in range(10)]
    x = [torch.rand(10, 1) for _ in range(10)]
    AxFn = torchopt.linalg.utils.normalize_matvec(A)
    Ax = AxFn(x)
    for Ax_item, A_item, x_item in zip(Ax, A, x):
        assert torch.equal(Ax_item, A_item @ x_item)
