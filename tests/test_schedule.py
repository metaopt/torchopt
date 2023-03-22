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

from __future__ import annotations

from typing import Any, Callable

import functorch
import numpy as np
import torch
import torch.nn.functional as F

import helpers
import torchopt
from torchopt.alias.utils import _set_use_chain_flat


@helpers.parametrize(
    init_value=[1.0, 1e-1],
    decay_rate=[1e-2, 1e-3],
    transition_begin=[1, 5],
    transition_steps=[10, 100],
    staircase=[False, True],
    end_value=[0.0, None, 8e-1],
)
def test_exponential_decay(
    init_value: float,
    decay_rate: float,
    transition_begin: int,
    transition_steps: int | None,
    staircase: bool,
    end_value: float | None,
) -> None:
    schedule = torchopt.schedule.exponential_decay(
        init_value=init_value,
        decay_rate=decay_rate,
        transition_steps=transition_steps,
        transition_begin=transition_begin,
        staircase=staircase,
        end_value=end_value,
    )
    if end_value is not None:
        clip_fn = max if decay_rate < 1.0 else min
    for i in range(transition_begin, transition_steps):
        lr = schedule(i)
        if staircase:
            lr_gt = init_value * (decay_rate ** np.floor((i - transition_begin) / transition_steps))
        else:
            lr_gt = init_value * (decay_rate ** ((i - transition_begin) / transition_steps))
        if end_value is not None:
            lr_gt = clip_fn(lr_gt, end_value)
        assert np.allclose(lr, lr_gt)


def test_linear_schedule() -> None:
    init_value = 1.0
    end_value = 0.0
    gap_value = init_value - end_value
    transition_steps = 10
    transition_begin = 1

    schedule = torchopt.schedule.linear_schedule(
        init_value=init_value,
        end_value=end_value,
        transition_steps=transition_steps,
        transition_begin=transition_begin,
    )
    for i in range(transition_begin, transition_steps):
        lr = schedule(i)
        lr_gt = init_value - gap_value * (i - transition_begin) / transition_steps
        assert np.allclose(lr, lr_gt)


@helpers.parametrize(
    dtype=[torch.float64, torch.float32],
    lr=[1e-2, 1e-3],
    total_iters=[helpers.NUM_UPDATES, helpers.NUM_UPDATES * 2],
    optimizers=[
        (torchopt.sgd, torch.optim.SGD, {}),
        (torchopt.adam, torch.optim.Adam, {}),
        (torchopt.adamw, torch.optim.AdamW, {}),
        (torchopt.adagrad, torch.optim.Adagrad, {'eps': 1e-8}),
        (torchopt.rmsprop, torch.optim.RMSprop, {}),
    ],
    inplace=[True, False],
    weight_decay=[0.0, 1e-2],
    use_chain_flat=[True, False],
)
def test_lr_linear_schedule(
    dtype: torch.dtype,
    lr: float,
    total_iters: int,
    optimizers: tuple[Callable, torch.optim.Optimizer, dict[str, Any]],
    inplace: bool,
    weight_decay: float,
    use_chain_flat: bool,
) -> None:
    _set_use_chain_flat(use_chain_flat)

    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    torchopt_optimizer, torch_optimizer, optimizer_kwargs = optimizers

    fmodel, params, buffers = functorch.make_functional_with_buffers(model)
    optim = torchopt_optimizer(
        torchopt.schedule.linear_schedule(
            init_value=lr,
            end_value=0.1 * lr,
            transition_steps=total_iters,
            transition_begin=0,
        ),
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )
    optim_state = optim.init(params)
    optim_ref = torch_optimizer(
        model_ref.parameters(),
        lr,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )
    torch_scheduler = torch.optim.lr_scheduler.LinearLR(
        optim_ref,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_iters,
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
        torch_scheduler.step()

    helpers.assert_model_all_close((params, buffers), model_ref, model_base, dtype=dtype)
    _set_use_chain_flat(True)
