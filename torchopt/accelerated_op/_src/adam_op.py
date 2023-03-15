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
"""The Python implementation of accelerated AdamOp."""

# pylint: disable=invalid-name,too-many-arguments,unused-argument

from __future__ import annotations

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adam forward inplace."""
    mu = mu.mul_(b1).add_(updates, alpha=1.0 - b1)
    nu = nu.mul_(b2).addcmul_(updates, updates, value=1.0 - b2)
    updates.copy_(
        mu.div(1.0 - pow(b1, count)).div_(
            nu.div(1.0 - pow(b2, count)).add_(eps_root).sqrt_().add_(eps),
        ),
    )
    return updates, mu, nu


def forward_mu(
    updates: torch.Tensor,
    mu: torch.Tensor,
    b1: float,
) -> torch.Tensor:
    """Adam forward mu."""
    return mu.mul(b1).add_(updates, alpha=1.0 - b1)


def forward_nu(
    updates: torch.Tensor,
    nu: torch.Tensor,
    b2: float,
) -> torch.Tensor:
    """Adam forward nu."""
    return nu.mul(b2).addcmul_(updates, updates, value=1.0 - b2)


def forward_updates(
    new_mu: torch.Tensor,
    new_nu: torch.Tensor,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    count: int,
) -> torch.Tensor:
    """Adam forward updates."""
    return new_mu.div(1.0 - pow(b1, count)).div_(
        new_nu.div(1.0 - pow(b2, count)).add_(eps_root).sqrt_().add_(eps),
    )


def backward_mu(
    dmu: torch.Tensor,
    updates: torch.Tensor,
    mu: torch.Tensor,
    b1: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adam backward mu."""
    dupdates = dmu.mul(1.0 - b1)
    dmu = dmu.mul(b1)
    return dupdates, dmu


def backward_nu(
    dnu: torch.Tensor,
    updates: torch.Tensor,
    nu: torch.Tensor,
    b2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adam backward nu."""
    dupdates = updates.mul(dnu).mul_(2.0 * (1.0 - b2))
    dnu = dnu.mul(b2)
    return dupdates, dnu


def backward_updates(
    dupdates: torch.Tensor,
    updates: torch.Tensor,
    new_mu: torch.Tensor,
    new_nu: torch.Tensor,
    b1: float,
    b2: float,
    eps_root: float,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adam backward updates."""
    one_minus_pow_b1 = 1.0 - pow(b1, count)
    inv_one_minus_pow_b2 = 1.0 / (1.0 - pow(b2, count) + eps_root)

    updates_div_new_mu = updates.div(new_mu)
    dnew_mu_out = dupdates.mul(updates_div_new_mu)
    denominator = updates_div_new_mu.mul_(one_minus_pow_b1)
    dnew_nu_out = (
        denominator.square_().mul_(dupdates).mul_(updates).mul_(-0.5 * inv_one_minus_pow_b2)
    )

    mask = new_mu == 0
    dnew_mu_out[mask] = 0
    dnew_nu_out[mask] = 0
    return dnew_mu_out, dnew_nu_out
