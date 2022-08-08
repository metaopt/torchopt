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

import copy
import itertools
import random
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils import data
from torchvision import models

import torchopt


def parametrize(**argvalues) -> pytest.mark.parametrize:
    arguments = tuple(argvalues)
    argvalues = tuple(itertools.product(*tuple(map(argvalues.get, arguments))))
    ids = tuple(
        '-'.join(f'{arg}({val})' for arg, val in zip(arguments, values)) for values in argvalues
    )

    return pytest.mark.parametrize(arguments, argvalues, ids=ids)


def get_models(
    device: Optional[Union[str, torch.device]] = None, dtype: torch.dtype = torch.float32
) -> Tuple[nn.Module, nn.Module, data.DataLoader]:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model = models.resnet18().to(dtype=dtype)
    model_ref = copy.deepcopy(model)
    if device is not None:
        model = model.to(device=torch.device(device))
        model_ref = model_ref.to(device=torch.device(device))

    batch_size = 8
    dataset = data.TensorDataset(
        torch.randn(batch_size * 2, 3, 224, 224), torch.randint(0, 1000, (batch_size * 2,))
    )
    loader = data.DataLoader(dataset, batch_size, shuffle=False)

    return model, model_ref, loader


@parametrize(
    dtype=[torch.float32, torch.float64],
    max_norm=[1.0, 10.0],
    lr=[1e-3, 1e-4, 1e-5],
    momentum=[0.0, 0.1, 0.2],
    nesterov=[False, True],
)
def test_sgd(
    dtype: torch.dtype, max_norm: float, lr: float, momentum: float, nesterov: bool
) -> None:
    model, model_ref, loader = get_models(device='cpu', dtype=dtype)

    chain = torchopt.combine.chain(
        torchopt.clip.clip_grad_norm(max_norm=max_norm),
        torchopt.sgd(lr=lr, momentum=(momentum if momentum != 0.0 else None), nesterov=nesterov),
    )
    optim = torchopt.Optimizer(model.parameters(), chain)
    optim_ref = torch.optim.SGD(
        model_ref.parameters(), lr, momentum=momentum, nesterov=nesterov, weight_decay=0.0
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
            assert torch.allclose(p, p_ref), f'{p!r} != {p_ref!r}'
        for b, b_ref in zip(model.buffers(), model_ref.buffers()):
            b = b.to(dtype=dtype) if not b.is_floating_point() else b
            b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
            assert torch.allclose(b, b_ref), f'{b!r} != {b_ref!r}'
