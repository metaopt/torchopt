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
import os
import random
from typing import Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils import data


BATCH_SIZE = 4
NUM_UPDATES = 3

MODEL_NUM_INPUTS = 28 * 28  # MNIST
MODEL_NUM_CLASSES = 10
MODEL_HIDDEN_SIZE = 64


def parametrize(**argvalues) -> pytest.mark.parametrize:
    arguments = list(argvalues)

    if 'dtype' in argvalues:
        dtypes = argvalues['dtype']
        argvalues['dtype'] = dtypes[:1]
        arguments.remove('dtype')
        arguments.insert(0, 'dtype')

        argvalues = list(itertools.product(*tuple(map(argvalues.get, arguments))))
        first_product = argvalues[0]
        argvalues.extend((dtype,) + first_product[1:] for dtype in dtypes[1:])

    ids = tuple(
        '-'.join(f'{arg}({val})' for arg, val in zip(arguments, values)) for values in argvalues
    )

    return pytest.mark.parametrize(arguments, argvalues, ids=ids)


def seed_everything(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass


@torch.no_grad()
def get_models(
    device: Optional[Union[str, torch.device]] = None, dtype: torch.dtype = torch.float32
) -> Tuple[nn.Module, nn.Module, nn.Module, data.DataLoader]:
    seed_everything(seed=42)

    model_base = nn.Sequential(
        nn.Linear(
            in_features=MODEL_NUM_INPUTS, out_features=MODEL_HIDDEN_SIZE, bias=True, dtype=dtype
        ),
        nn.ReLU(),
        nn.Linear(
            in_features=MODEL_HIDDEN_SIZE, out_features=MODEL_HIDDEN_SIZE, bias=True, dtype=dtype
        ),
        nn.ReLU(),
        nn.Linear(
            in_features=MODEL_HIDDEN_SIZE, out_features=MODEL_NUM_CLASSES, bias=True, dtype=dtype
        ),
        nn.Softmax(),
    )
    for name, param in model_base.named_parameters(recurse=True):
        if name.endswith('weight'):
            nn.init.orthogonal_(param)
        if name.endswith('bias'):
            param.data.normal_(0, 0.1)

    model = copy.deepcopy(model_base)
    model_ref = copy.deepcopy(model_base)
    if device is not None:
        model_base = model_base.to(device=torch.device(device))
        model = model.to(device=torch.device(device))
        model_ref = model_ref.to(device=torch.device(device))

    dataset = data.TensorDataset(
        torch.randint(0, 1, (BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS)),
        torch.randint(0, MODEL_NUM_CLASSES, (BATCH_SIZE * NUM_UPDATES,)),
    )
    loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=False)

    return model, model_ref, model_base, loader


@torch.no_grad()
def assert_model_all_close(
    model: nn.Module,
    model_ref: nn.Module,
    model_base: nn.Module,
    dtype: torch.dtype = torch.float32,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
):

    for p, p_ref, p_base in zip(
        model.parameters(), model_ref.parameters(), model_base.parameters()
    ):
        assert_all_close(p, p_ref, base=p_base, rtol=rtol, atol=atol, equal_nan=equal_nan)
    for b, b_ref, b_base in zip(model.buffers(), model_ref.buffers(), model_base.buffers()):
        b = b.to(dtype=dtype) if not b.is_floating_point() else b
        b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
        b_base = b_base.to(dtype=dtype) if not b_base.is_floating_point() else b_base
        assert_all_close(b, b_ref, base=b_base, rtol=rtol, atol=atol, equal_nan=equal_nan)


@torch.no_grad()
def assert_all_close(
    input: torch.Tensor,
    other: torch.Tensor,
    base: torch.Tensor = None,
    dtype: torch.dtype = torch.float32,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
) -> None:

    finfo = torch.finfo(dtype)

    if rtol is None:
        rtol = max(1e-05, 2 * NUM_UPDATES * finfo.eps)
    if atol is None:
        atol = max(1e-08, 2 * NUM_UPDATES * finfo.resolution)

    if base is not None:
        input = input - base
        other = other - base

    assert torch.allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan), (
        f'L_inf = {(input - other).abs().max():.5e}, '
        f'fail_rate = {torch.logical_not((input - other).abs() <= atol + rtol * other.abs()).float().mean()} '
        f'(atol = {atol:.5e}, rtol = {rtol:.5e})'
    )
