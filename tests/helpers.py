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
from torchvision import models


BATCH_SIZE = 4
NUM_UPDATES = 3


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


def get_models(
    device: Optional[Union[str, torch.device]] = None, dtype: torch.dtype = torch.float32
) -> Tuple[nn.Module, nn.Module, data.DataLoader]:
    seed_everything(seed=42)

    model = models.resnet18().to(dtype=dtype)
    model_ref = copy.deepcopy(model)
    if device is not None:
        model = model.to(device=torch.device(device))
        model_ref = model_ref.to(device=torch.device(device))

    dataset = data.TensorDataset(
        torch.randn(BATCH_SIZE, 3, 224, 224), torch.randint(0, 1000, (BATCH_SIZE,))
    )
    loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=False)

    return model, model_ref, loader


def assert_all_close(
    input: torch.Tensor,
    other: torch.Tensor,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> None:

    if dtype is None:
        dtype = input.dtype
    finfo = torch.finfo(input.dtype)

    if rtol is None:
        rtol = 2.0 * finfo.eps
    if atol is None:
        atol = 2.0 * finfo.resolution

    assert torch.allclose(input, other, rtol=rtol, atol=atol, equal_nan=equal_nan), (
        f'L_inf = {(input - other).abs().max()}, '
        f'fail_rate = {torch.logical_not((input - other).abs() <= atol + rtol * other.abs()).float().mean()}'
    )
