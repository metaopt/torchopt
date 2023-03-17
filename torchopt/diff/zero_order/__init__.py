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
"""Zero-Order Gradient."""

import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any, Callable

import torch

from torchopt.diff.zero_order import nn
from torchopt.diff.zero_order.decorator import zero_order
from torchopt.diff.zero_order.nn import ZeroOrderGradientModule


__all__ = ['zero_order', 'ZeroOrderGradientModule']


class _CallableModule(_ModuleType):  # pylint: disable=too-few-public-methods
    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
        return self.zero_order(*args, **kwargs)


# Replace entry in sys.modules for this module with an instance of _CallableModule
_modself = _sys.modules[__name__]
_modself.__class__ = _CallableModule
del _sys, _ModuleType, _modself, _CallableModule
