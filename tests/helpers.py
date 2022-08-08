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


def parametrize(**argvalues) -> pytest.mark.parametrize:
    arguments = tuple(argvalues)
    argvalues = tuple(itertools.product(*tuple(map(argvalues.get, arguments))))
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

    batch_size = 8
    dataset = data.TensorDataset(
        torch.randn(batch_size * 2, 3, 224, 224), torch.randint(0, 1000, (batch_size * 2,))
    )
    loader = data.DataLoader(dataset, batch_size, shuffle=False)

    return model, model_ref, loader
