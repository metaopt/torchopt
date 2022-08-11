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

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import helpers
import torchopt


@helpers.parametrize(
    dtype=[torch.float32, torch.float64],
    max_norm=[1.0, 10.0],
    lr=[1e-3, 1e-4],
    momentum=[0.0, 0.1],
    nesterov=[False, True],
)
def test_sgd(
    dtype: torch.dtype, max_norm: float, lr: float, momentum: float, nesterov: bool
) -> None:
    if nesterov and momentum <= 0.0:
        pytest.skip('Nesterov momentum requires a momentum and zero dampening.')

    model, model_ref, loader = helpers.get_models(device='cpu', dtype=dtype)

    chain = torchopt.combine.chain(
        torchopt.clip.clip_grad_norm(max_norm=max_norm),
        torchopt.sgd(lr=lr, momentum=(momentum if momentum != 0.0 else None), nesterov=nesterov),
    )
    optim = torchopt.Optimizer(model.parameters(), chain)
    optim_ref = torch.optim.SGD(
        model_ref.parameters(),
        lr,
        momentum=momentum,
        dampening=0.0,
        nesterov=nesterov,
        weight_decay=0.0,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = model(xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        optim.zero_grad()
        loss.backward()
        optim.step()

        optim_ref.zero_grad()
        loss_ref.backward()
        clip_grad_norm_(model_ref.parameters(), max_norm=max_norm)
        optim_ref.step()

    with torch.no_grad():
        for p, p_ref in zip(model.parameters(), model_ref.parameters()):
            helpers.assert_all_close(p, p_ref)
        for b, b_ref in zip(model.buffers(), model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            helpers.assert_all_close(b, b_ref)
