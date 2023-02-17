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


def test_nan_to_num() -> None:
    fn = torchopt.nan_to_num(0.0, 1.0, -1.0)
    nan = torch.tensor(torch.nan)
    inf = torch.tensor(torch.inf)
    ninf = torch.tensor(-torch.inf)
    updated, _ = fn.update(nan, None, inplace=False)
    assert torch.equal(updated, torch.tensor(0.0))
    assert updated is not nan

    updated, _ = fn.update(inf, None, inplace=False)
    assert torch.equal(updated, torch.tensor(1.0))
    assert updated is not inf

    updated, _ = fn.update(ninf, None, inplace=False)
    assert torch.equal(updated, torch.tensor(-1.0))
    assert updated is not ninf

    updated, _ = fn.update(nan, None, inplace=True)
    assert torch.equal(updated, torch.tensor(0.0))
    assert updated is nan

    updated, _ = fn.update(inf, None, inplace=True)
    assert torch.equal(updated, torch.tensor(1.0))
    assert updated is inf

    updated, _ = fn.update(ninf, None, inplace=True)
    assert torch.equal(updated, torch.tensor(-1.0))
    assert updated is ninf


def test_masked() -> None:
    fn = torchopt.nan_to_num(0.0, 1.0, -1.0)
    nan = torch.tensor(torch.nan)
    updates = [nan, nan, nan]

    masked_fn = torchopt.transform.masked(fn, [True, False, True])
    state = masked_fn.init(updates)

    updates, _ = masked_fn.update(updates, state)
    assert nan is updates[1]
