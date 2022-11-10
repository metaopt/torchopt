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
"""The base class for zero-order gradient models."""

import abc
from typing import Sequence, Type, Union

import torch
import torch.nn as nn

from torchopt.diff.zero_order.decorator import Samplable, zero_order
from torchopt.typing import Numeric


__all__ = ['ZeroOrderGradientModule']


def enable_zero_order_gradients(
    cls: Type['ZeroOrderGradientModule'], *args, **kwargs
) -> Type['ZeroOrderGradientModule']:
    """Enables zero-order gradient estimation for the :func:`forward` method."""
    if getattr(cls.forward, '__zero_order_gradients_enabled__', False):
        raise TypeError(
            'Zero-order gradient estimation is already enabled for the `forward` method.'
        )

    def distribution_fn(*args, **kwargs):  # pylint: disable=unused-argument
        return cls.sample  # FIX: signature

    wrapped = zero_order(distribution_fn, *args, **kwargs)(cls.forward)
    wrapped.__zero_order_gradients_enabled__ = True  # type: ignore[attr-defined]
    cls.forward = wrapped  # type: ignore[assignment]
    return cls


class ZeroOrderGradientModule(nn.Module, Samplable):
    """The base class for zero-order gradient models."""

    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__()
        enable_zero_order_gradients(cls, *args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """The forward pass of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
        self, sample_shape: torch.Size = torch.Size()  # pylint: disable=unused-argument
    ) -> Union[torch.Tensor, Sequence[Numeric]]:
        # pylint: disable-next=line-too-long
        """Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched."""
        raise NotImplementedError
