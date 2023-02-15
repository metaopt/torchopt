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
from torchopt import pytree


def test_nan_to_num_hook() -> None:
    nan = torch.tensor(torch.nan)
    inf = torch.tensor(torch.inf)
    ninf = torch.tensor(-torch.inf)
    hook = torchopt.hook.nan_to_num_hook(0.0, 1.0, -1.0)
    result = pytree.tree_map(hook, [nan, inf, ninf])
    assert torch.equal(result[0], torch.tensor(0.0))
    assert torch.equal(result[1], torch.tensor(1.0))
    assert torch.equal(result[2], torch.tensor(-1.0))


def test_zero_nan_hook() -> None:
    tensor = torch.tensor(1.0, requires_grad=True)
    hook = torchopt.hook.zero_nan_hook
    fn = torchopt.register_hook(hook)
    fn.update(tensor, None)
    assert tensor._backward_hooks[0] is hook
