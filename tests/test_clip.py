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

import pytest
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import helpers
import torchopt
from torchopt.alias.utils import _set_use_chain_flat


@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    max_norm=[1.0, 10.0],
    lr=[1e-2, 1e-3, 1e-4],
    momentum=[0.0, 0.1],
    dampening=[0.0, 0.5],
    nesterov=[False, True],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_chain_flat=[True, False],
)
def test_sgd(
    dtype: torch.dtype,
    max_norm: float,
    lr: float,
    momentum: float,
    dampening: float,
    nesterov: bool,
    weight_decay: float,
    maximize: bool,
    use_chain_flat: bool,
) -> None:
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        pytest.skip('Nesterov momentum requires a momentum and zero dampening.')

    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    chain = torchopt.chain(
        torchopt.clip.clip_grad_norm(max_norm=max_norm),
        torchopt.sgd(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            weight_decay=weight_decay,
            maximize=maximize,
        ),
    )
    optim = torchopt.Optimizer(model.parameters(), chain)
    optim_ref = torch.optim.SGD(
        model_ref.parameters(),
        lr,
        momentum=momentum,
        dampening=dampening,
        nesterov=nesterov,
        weight_decay=weight_decay,
        maximize=maximize,
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

    helpers.assert_model_all_close(model, model_ref, model_base, dtype=dtype)
    _set_use_chain_flat(True)
