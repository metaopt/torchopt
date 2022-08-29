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
from typing import Iterable, Optional, Tuple, Union
import types
from collections import OrderedDict

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils import data

import jax
import jax.numpy as jnp



BATCH_SIZE = 4
NUM_UPDATES = 3

MODEL_NUM_INPUTS = 10
MODEL_NUM_CLASSES = 10
#MODEL_HIDDEN_SIZE = 64


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


def get_model_jax(
    dtype: np.dtype = np.float32
) -> Tuple[types.FunctionType, OrderedDict]:
    seed_everything(seed=42)
    def f(params, x):
        return jnp.matmul(x, params['weight']) + params['bias']
    p = OrderedDict([('weight', jnp.ones((MODEL_NUM_INPUTS, MODEL_NUM_CLASSES))), ('bias', jnp.zeros((MODEL_NUM_CLASSES,)))])
    return f, p

@torch.no_grad()
def get_model_torch(
    device: Optional[Union[str, torch.device]] = None, dtype: torch.dtype = torch.float32
) -> Tuple[nn.Module, data.DataLoader]:
    seed_everything(seed=42)

    class Net(nn.Module):
        def __init__(self, dim, out):
            super().__init__()
            self.fc = nn.Linear(dim, out, bias=True)
            nn.init.ones_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

        def forward(self, x):
            return self.fc(x)
    
    model = Net(MODEL_NUM_INPUTS, MODEL_NUM_CLASSES)

    if device is not None:
        model = model.to(device=torch.device(device))
        
    dataset = data.TensorDataset(
        torch.randint(0, 1, (BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS)),
        torch.randint(0, MODEL_NUM_CLASSES, (BATCH_SIZE * NUM_UPDATES,)),
    )
    loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=False)

    return model, loader

def clone(p):
    p_out = []
    for item in p:
        if isinstance(item, torch.Tensor):
            p_out.append(item.clone().detach_().requires_grad_(True))
        else:
            p_out.append(item)
    return tuple(p_out)

@torch.no_grad()
def assert_model_all_close(
    model: Union[nn.Module, Tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]]],
    model_ref: nn.Module,
    model_base: nn.Module,
    dtype: torch.dtype = torch.float32,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
):

    if isinstance(model, tuple):
        params, buffers = model
    elif isinstance(model, nn.Module):
        params = model.parameters()
        buffers = model.buffers()

    for p, p_ref, p_base in zip(params, model_ref.parameters(), model_base.parameters()):
        assert_all_close(p, p_ref, base=p_base, rtol=rtol, atol=atol, equal_nan=equal_nan)
    for b, b_ref, b_base in zip(buffers, model_ref.buffers(), model_base.buffers()):
        b = b.to(dtype=dtype) if not b.is_floating_point() else b
        b_ref = b_ref.to(dtype=dtype) if not b_ref.is_floating_point() else b_ref
        b_base = b_base.to(dtype=dtype) if not b_base.is_floating_point() else b_base
        assert_all_close(b, b_ref, base=b_base, rtol=rtol, atol=atol, equal_nan=equal_nan)


@torch.no_grad()
def assert_all_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    base: torch.Tensor = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
) -> None:

    if base is not None:
        actual = actual - base
        expected = expected - base

    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_dtype=True,
    )