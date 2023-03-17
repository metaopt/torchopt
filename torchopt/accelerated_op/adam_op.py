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
"""The accelerated AdamOp."""

# pylint: disable=c-extension-no-member,invalid-name

from __future__ import annotations

import contextlib
from typing import Any

import torch


try:
    from torchopt._C import adam_op  # pylint: disable=no-name-in-module
except ImportError:
    from torchopt.accelerated_op._src import adam_op  # type: ignore[no-redef]


class AdamOp:  # pylint: disable=too-few-public-methods
    """Fused accelerated Adam operators."""

    class MuOp(torch.autograd.Function):  # pylint: disable=abstract-method
        """Bias-corrected first moment estimate."""

        @staticmethod
        def jvp(ctx: Any, *grad_inputs: Any) -> Any:
            """Define a formula for differentiating the operation with forward mode automatic differentiation."""

        @staticmethod
        def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            """Perform the operation."""
            updates, mu, b1 = args
            new_mu = adam_op.forward_mu(updates, mu, b1)
            ctx.save_for_backward(updates, mu)
            ctx.b1 = b1
            return new_mu

        @staticmethod
        def backward(ctx: Any, *args: Any) -> Any:
            # pylint: disable-next=line-too-long
            """Define a formula for differentiating the operation with backward mode automatic differentiation (alias to the :meth:`vjp` method)."""
            dmu = args[0]
            updates, mu = ctx.saved_tensors
            b1 = ctx.b1
            result = adam_op.backward_mu(dmu, updates, mu, b1)
            return result[0], result[1], None

    class NuOp(torch.autograd.Function):  # pylint: disable=abstract-method
        """Bias-corrected second raw moment estimate."""

        @staticmethod
        def jvp(ctx: Any, *grad_inputs: Any) -> Any:
            """Define a formula for differentiating the operation with forward mode automatic differentiation."""

        @staticmethod
        def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            """Perform the operation."""
            updates, nu, b2 = args
            new_nu = adam_op.forward_nu(updates, nu, b2)
            ctx.save_for_backward(updates, nu)
            ctx.b2 = b2
            return new_nu

        @staticmethod
        def backward(ctx: Any, *args: Any) -> Any:
            # pylint: disable-next=line-too-long
            """Define a formula for differentiating the operation with backward mode automatic differentiation (alias to the :meth:`vjp` function)."""
            dnu = args[0]
            updates, nu = ctx.saved_tensors
            b2 = ctx.b2
            result = adam_op.backward_nu(dnu, updates, nu, b2)
            return result[0], result[1], None

    class UpdatesOp(torch.autograd.Function):  # pylint: disable=abstract-method
        """Adam updates."""

        @staticmethod
        def jvp(ctx: Any, *grad_inputs: Any) -> Any:
            """Define a formula for differentiating the operation with forward mode automatic differentiation."""

        @staticmethod
        def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            """Perform the operation."""
            new_mu, new_nu, (b1, b2, eps, eps_root, count) = args
            new_updates = adam_op.forward_updates(new_mu, new_nu, b1, b2, eps, eps_root, count)
            ctx.save_for_backward(new_updates, new_mu, new_nu)
            ctx.others = (b1, b2, eps, eps_root, count)
            return new_updates

        @staticmethod
        def backward(ctx: Any, *args: Any) -> Any:
            # pylint: disable-next=line-too-long
            """Define a formula for differentiating the operation with backward mode automatic differentiation (alias to the :meth:`vjp` function)."""
            dupdates = args[0]
            updates, new_mu, new_nu = ctx.saved_tensors
            b1, b2, _, eps_root, count = ctx.others
            result = adam_op.backward_updates(
                dupdates,
                updates,
                new_mu,
                new_nu,
                b1,
                b2,
                eps_root,
                count,
            )
            return result[0], result[1], None

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        *,
        eps_root: float = 0.0,
        inplace: bool = True,
    ) -> None:
        """Initialize the Adam operator."""
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.eps_root = eps_root
        self.inplace = inplace

    def __call__(
        self,
        mu: torch.Tensor,
        nu: torch.Tensor,
        updates: torch.Tensor | None,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply the Adam operator."""
        if updates is None:
            return mu, nu, None
        device_context = (
            torch.cuda.device(torch.cuda.current_device())
            if updates.is_cuda
            else contextlib.nullcontext()
        )
        with device_context:  # type: ignore[attr-defined]
            if self.inplace:
                new_updates, new_mu, new_nu = adam_op.forward_(
                    updates,
                    mu,
                    nu,
                    self.b1,
                    self.b2,
                    self.eps,
                    self.eps_root,
                    count,
                )
            else:
                new_mu = self.MuOp.apply(updates, mu, self.b1)
                new_nu = self.NuOp.apply(updates, nu, self.b2)
                new_updates = self.UpdatesOp.apply(
                    new_mu,
                    new_nu,
                    (
                        self.b1,
                        self.b2,
                        self.eps,
                        self.eps_root,
                        count,
                    ),
                )
        return new_mu, new_nu, new_updates
