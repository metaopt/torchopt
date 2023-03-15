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

# pylint: disable=all

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
def forward_mu(updates: torch.Tensor, mu: torch.Tensor, b1: float) -> torch.Tensor: ...
def forward_nu(updates: torch.Tensor, nu: torch.Tensor, b2: float) -> torch.Tensor: ...
def forward_updates(
    new_mu: torch.Tensor,
    new_nu: torch.Tensor,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    count: int,
) -> torch.Tensor: ...
def backward_mu(
    dmu: torch.Tensor,
    updates: torch.Tensor,
    mu: torch.Tensor,
    b1: float,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def backward_nu(
    dnu: torch.Tensor,
    updates: torch.Tensor,
    nu: torch.Tensor,
    b2: float,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def backward_updates(
    dupdates: torch.Tensor,
    updates: torch.Tensor,
    new_mu: torch.Tensor,
    new_nu: torch.Tensor,
    b1: float,
    b2: float,
    eps_root: float,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
