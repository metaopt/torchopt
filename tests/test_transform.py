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

from typing import Tuple

import functorch
import torch
import torch.nn.functional as F

import helpers
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


@helpers.parametrize(
    dtype=[torch.float64],
    lr=[1e-2, 1e-3, 1e-4],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
)
def test_scale_by_adam(
    dtype: torch.dtype,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    inplace: bool,
    weight_decay: float,
    maximize: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt.adam(
        lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
    )
    optim_state = optim.init(params)
    optim_ref = torch.optim.Adam(
        model_ref.parameters(),
        lr,
        betas=betas,
        eps=eps,
        amsgrad=False,
        weight_decay=weight_decay,
        maximize=maximize,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)
        pred = fmodel(params, buffers, xs)
        pred_ref = model_ref(xs)
        loss = F.cross_entropy(pred, ys)
        loss_ref = F.cross_entropy(pred_ref, ys)

        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates, optim_state = optim.update(grads, optim_state, params=params, inplace=inplace)
        params = torchopt.apply_updates(params, updates, inplace=inplace)

        optim_ref.zero_grad()
        loss_ref.backward()
        optim_ref.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)
