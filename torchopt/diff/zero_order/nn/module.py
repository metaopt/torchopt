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

import abc
import functools
from typing import Dict, Optional, Sequence, Tuple, Type, Union

import torch
import torch.nn as nn

from torchopt import pytree
from torchopt.diff.implicit.nn.module import container_context
from torchopt.diff.zero_order.decorator import Method, Samplable, zero_order
from torchopt.typing import Numeric, TupleOfTensors
from torchopt.utils import extract_module_containers


__all__ = ['ZeroOrderGradientModule']


def enable_zero_order_gradients(
    cls: Type['ZeroOrderGradientModule'],
    method: Method = 'naive',
    num_samples: int = 1,
    sigma: Numeric = 1.0,
) -> Type['ZeroOrderGradientModule']:
    """Enable zero-order gradient estimation for the :func:`forward` method."""
    cls_forward = cls.forward
    if getattr(cls_forward, '__zero_order_gradients_enabled__', False):
        raise TypeError(
            'Zero-order gradient estimation is already enabled for the `forward` method.'
        )

    @functools.wraps(cls_forward)
    def wrapped(  # pylint: disable=too-many-locals
        self: 'ZeroOrderGradientModule', *input, **kwargs
    ) -> torch.Tensor:
        """Do the forward pass calculation."""
        params_containers = extract_module_containers(self, with_buffers=False)[0]

        flat_params: TupleOfTensors
        flat_params, params_containers_treespec = pytree.tree_flatten_as_tuple(
            params_containers  # type: ignore[arg-type]
        )

        @zero_order(self.sample, argnums=0, method=method, num_samples=num_samples, sigma=sigma)
        def forward_fn(
            __flat_params: TupleOfTensors,  # pylint: disable=unused-argument
            *input,
            **kwargs,
        ) -> torch.Tensor:
            flat_grad_tracking_params = __flat_params
            grad_tracking_params_containers: Tuple[
                Dict[str, Optional[torch.Tensor]], ...
            ] = pytree.tree_unflatten(  # type: ignore[assignment]
                params_containers_treespec, flat_grad_tracking_params
            )

            with container_context(
                params_containers,
                grad_tracking_params_containers,
            ):
                return cls_forward(self, *input, **kwargs)

        return forward_fn(flat_params, *input, **kwargs)

    wrapped.__zero_order_gradients_enabled__ = True  # type: ignore[attr-defined]
    cls.forward = wrapped  # type: ignore[assignment]
    return cls


class ZeroOrderGradientModule(nn.Module, Samplable):
    """The base class for zero-order gradient models."""

    def __init_subclass__(  # pylint: disable=arguments-differ
        cls,
        method: Method = 'naive',
        num_samples: int = 1,
        sigma: Numeric = 1.0,
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
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Do the forward pass of the model."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
        self, sample_shape: torch.Size = torch.Size()  # pylint: disable=unused-argument
    ) -> Union[torch.Tensor, Sequence[Numeric]]:
        # pylint: disable-next=line-too-long
        """Generate a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched."""
        raise NotImplementedError
