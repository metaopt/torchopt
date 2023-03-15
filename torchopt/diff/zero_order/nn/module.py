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
"""The base class for zero-order gradient models."""

# pylint: disable=redefined-builtin

from __future__ import annotations

import abc
import functools
from typing import Any, Sequence

import torch
import torch.nn as nn

from torchopt.diff.zero_order.decorator import Method, Samplable, zero_order
from torchopt.nn.stateless import reparametrize
from torchopt.typing import Numeric, TupleOfTensors


__all__ = ['ZeroOrderGradientModule']


def enable_zero_order_gradients(
    cls: type[ZeroOrderGradientModule],
    method: Method = 'naive',
    num_samples: int = 1,
    sigma: float = 1.0,
) -> type[ZeroOrderGradientModule]:
    """Enable zero-order gradient estimation for the :func:`forward` method."""
    cls_forward = cls.forward
    if getattr(cls_forward, '__zero_order_gradients_enabled__', False):
        raise TypeError(
            'Zero-order gradient estimation is already enabled for the `forward` method.',
        )

    @functools.wraps(cls_forward)
    def wrapped(self: ZeroOrderGradientModule, *input: Any, **kwargs: Any) -> torch.Tensor:
        """Do the forward pass calculation."""
        named_params = tuple(self.named_parameters())
        if len(named_params) == 0:
            raise RuntimeError('The module has no parameters.')
        params_names, flat_params = tuple(zip(*named_params))

        @zero_order(self.sample, argnums=0, method=method, num_samples=num_samples, sigma=sigma)
        def forward_fn(
            __flat_params: TupleOfTensors,
            *input: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            with reparametrize(self, zip(params_names, __flat_params)):
                return cls_forward(self, *input, **kwargs)

        return forward_fn(flat_params, *input, **kwargs)

    wrapped.__zero_order_gradients_enabled__ = True  # type: ignore[attr-defined]
    cls.forward = wrapped  # type: ignore[method-assign]
    return cls


class ZeroOrderGradientModule(nn.Module, Samplable):
    """The base class for zero-order gradient models."""

    def __init_subclass__(  # pylint: disable=arguments-differ
        cls,
        method: Method = 'naive',
        num_samples: int = 1,
        sigma: float = 1.0,
    ) -> None:
        """Validate and initialize the subclass."""
        super().__init_subclass__()
        enable_zero_order_gradients(
            cls,
            method=method,
            num_samples=num_samples,
            sigma=sigma,
        )

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Do the forward pass of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008 # pylint: disable=unused-argument
    ) -> torch.Tensor | Sequence[Numeric]:
        # pylint: disable-next=line-too-long
        """Generate a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched."""
        raise NotImplementedError
