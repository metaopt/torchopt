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

# isort: off

from typing import Tuple

import torch

def forward_(
    updates: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def forwardMu(updates: torch.Tensor, mu: torch.Tensor, b1: float) -> torch.Tensor: ...
def forwardNu(updates: torch.Tensor, nu: torch.Tensor, b2: float) -> torch.Tensor: ...
def forwardUpdates(
    new_mu: torch.Tensor,
    new_nu: torch.Tensor,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    count: int,
) -> torch.Tensor: ...
def backwardMu(
    dmu: torch.Tensor, updates: torch.Tensor, mu: torch.Tensor, b1: float
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def backwardNu(
    dnu: torch.Tensor, updates: torch.Tensor, nu: torch.Tensor, b2: float
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def backwardUpdates(
    dupdates: torch.Tensor,
    updates: torch.Tensor,
    new_mu: torch.Tensor,
    new_nu: torch.Tensor,
    b1: float,
    b2: float,
    count: int,
) -> Tuple[torch.Tensor, torch.Tensor]: ...
