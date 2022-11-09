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
"""The base class for zero order gradient models."""

from functools import partial
from typing import Type

from torchopt.diff.zero_order.decorator import zero_order


__all__ = ['ZeroOrderGradientModule']


def enable_zero_order_gradients(
    cls: Type['ZeroOrderGradientModule'], *args, **kwargs
) -> Type['ZeroOrderGradientModule']:
    cls.solve = zero_order(cls, *args, **kwargs)(cls.solve)


class ZeroOrderGradientModule:
    def __init_subclass__(cls, *args, **kwargs) -> None:
        super.__init_subclass__()
        enable_zero_order_gradients(cls, *args, **kwargs)
