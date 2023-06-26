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

import contextlib
import copy
import itertools
import os
import random
from typing import Iterable

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.types
from torch.utils import data

from torchopt import pytree
from torchopt.typing import TensorTree


BATCH_SIZE = 64
NUM_UPDATES = 5

MODEL_NUM_INPUTS = 28 * 28  # MNIST
MODEL_NUM_CLASSES = 10
MODEL_HIDDEN_SIZE = 64


def dtype_numpy2torch(dtype: np.dtype) -> torch.dtype:
    return torch.tensor(np.zeros(1, dtype=dtype)).dtype


def dtype_torch2numpy(dtype: torch.dtype) -> np.dtype:
    return torch.zeros(1, dtype=dtype).numpy().dtype


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
    else:
        argvalues = list(itertools.product(*tuple(map(argvalues.get, arguments))))

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
    with contextlib.suppress(AttributeError):
        torch.use_deterministic_algorithms(True)


class MyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.unused_module = nn.Linear(1, 1, bias=False)
        self.unused_parameter = nn.Parameter(torch.zeros(1, 1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@torch.no_grad()
def get_model():
    return nn.Sequential(
        MyLinear(
            in_features=MODEL_NUM_INPUTS,
            out_features=MODEL_HIDDEN_SIZE,
            bias=True,
        ),
        nn.BatchNorm1d(
            num_features=MODEL_HIDDEN_SIZE,
            track_running_stats=True,
        ),
        nn.ReLU(),
        nn.Linear(
            in_features=MODEL_HIDDEN_SIZE,
            out_features=MODEL_HIDDEN_SIZE,
            bias=True,
        ),
        nn.BatchNorm1d(
            num_features=MODEL_HIDDEN_SIZE,
            track_running_stats=True,
        ),
        nn.ReLU(),
        nn.Linear(
            in_features=MODEL_HIDDEN_SIZE,
            out_features=MODEL_NUM_CLASSES,
            bias=False,
        ),
        nn.Softmax(dim=-1),
    )


@torch.no_grad()
def get_models(
    device: torch.types.Device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[nn.Module, nn.Module, nn.Module, data.DataLoader]:
    seed_everything(seed=42)

    model_base = get_model().to(dtype=dtype)
    for name, param in model_base.named_parameters(recurse=True):
        if name.endswith('weight') and param.ndim >= 2:
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
        # torch.empty((BATCH_SIZE * NUM_UPDATES, MODEL_NUM_INPUTS), dtype=dtype).uniform_(-1.0, +1.0),
        torch.randint(0, MODEL_NUM_CLASSES, (BATCH_SIZE * NUM_UPDATES,)),
    )
    loader = data.DataLoader(dataset, BATCH_SIZE, shuffle=False)

    return model, model_ref, model_base, loader


@torch.no_grad()
def assert_model_all_close(
    model: nn.Module | tuple[Iterable[torch.Tensor], Iterable[torch.Tensor]],
    model_ref: nn.Module,
    model_base: nn.Module,
    dtype: torch.dtype = torch.float32,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> None:
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
    base: torch.Tensor | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> None:
    if base is not None:
        actual = actual - base
        expected = expected - base

    if rtol is None or atol is None:
        from torch.testing._comparison import get_tolerances

        rtol, atol = get_tolerances(actual, expected, rtol=rtol, atol=atol)
        rtol *= 5 * NUM_UPDATES
        atol *= 5 * NUM_UPDATES

    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_dtype=True,
    )


@torch.no_grad()
def assert_pytree_all_close(
    actual: TensorTree,
    expected: TensorTree,
    base: TensorTree | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
) -> None:
    actual_leaves, actual_treespec = pytree.tree_flatten(actual)
    expected_leaves, expected_treespec = pytree.tree_flatten(expected)
    assert actual_treespec == expected_treespec
    if base is not None:
        base_leaves, base_treespec = pytree.tree_flatten(base)
        assert base_treespec == expected_treespec
    else:
        base_leaves = [None] * len(actual_leaves)

    for actual_leaf, expected_leaf, base_leaf in zip(actual_leaves, expected_leaves, base_leaves):
        assert_all_close(
            actual_leaf,
            expected_leaf,
            base=base_leaf,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
