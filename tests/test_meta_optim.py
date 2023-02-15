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

import torch
import torch.nn.functional as F

import helpers
import torchopt


@helpers.parametrize(
    dtype=[torch.float64],
    outer_lr=[1e-2, 1e-3, 1e-4],
    inner_lr=[1e-2, 1e-3, 1e-4],
    inner_update=[2, 3, 5],
    betas=[(0.9, 0.999), (0.95, 0.9995)],
    eps=[1e-8],
    eps_root=[0.0, 1e-8],
    weight_decay=[0.0, 1e-2],
    maximize=[False, True],
    use_accelerated_op=[False, True],
    moment_requires_grad=[True, False],
)
def test_maml_meta_adam(
    dtype: torch.dtype,
    outer_lr: float,
    inner_lr: float,
    inner_update: int,
    betas: tuple[float, float],
    eps: float,
    eps_root: float,
    weight_decay: float,
    maximize: bool,
    use_accelerated_op: bool,
    moment_requires_grad: bool,
) -> None:
    model, model_ref, model_base, loader = helpers.get_models(device='cpu', dtype=dtype)

    outer_optim = torchopt.Adam(
        model.parameters(),
        outer_lr,
        betas=betas,
        eps=eps,
        eps_root=0.0,
        weight_decay=weight_decay,
        maximize=maximize,
        use_accelerated_op=use_accelerated_op,
    )

    for xs, ys in loader:
        xs = xs.to(dtype=dtype)

        inner_optim = torchopt.MetaAdam(
            module=model,
            lr=inner_lr,
            betas=betas,
            eps=eps,
            eps_root=eps_root,
            moment_requires_grad=moment_requires_grad,
            weight_decay=weight_decay,
            maximize=maximize,
            use_accelerated_op=use_accelerated_op,
        )

        for _ in range(inner_update):
            pred = model(xs)
            inner_loss = F.cross_entropy(pred, ys)  # compute loss
            inner_optim.step(inner_loss)

        pred = model(xs)
        outer_loss = F.cross_entropy(pred, ys)
        outer_optim.zero_grad()
        outer_loss.backward()
        outer_optim.step()

        torchopt.stop_gradient(model)
