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
"""Typing utilities."""

from typing import Callable, Optional, TypeVar, Union

import torch.distributed.rpc as rpc
from optree.typing import PyTree
from torch import Tensor
from torch.futures import Future

from torchopt.base import ChainedGradientTransformation, EmptyState, GradientTransformation


try:
    from typing import TypeAlias  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import TypeAlias


__all__ = [
    'GradientTransformation',
    'ChainedGradientTransformation',
    'EmptyState',
    'Params',
    'Updates',
    'OptState',
    'Scalar',
    'Numeric',
    'Schedule',
    'ScalarOrSchedule',
    'PyTree',
    'Tensor',
    'TensorTree',
    'OptionalTensorTree',
    'Future',
]

T = TypeVar('T')

Scalar: TypeAlias = Union[float, int, bool]
Numeric: TypeAlias = Union[Tensor, Scalar]

Schedule: TypeAlias = Callable[[Numeric], Numeric]
ScalarOrSchedule: TypeAlias = Union[float, Schedule]

TensorTree: TypeAlias = PyTree[Tensor]
OptionalTensorTree: TypeAlias = PyTree[Optional[Tensor]]

# Parameters are arbitrary nests of `torch.Tensor`.
Params: TypeAlias = TensorTree
Updates: TypeAlias = Params  # Gradient updates are of the same type as parameters.
OptState: TypeAlias = TensorTree  # States are arbitrary nests of `torch.Tensor`.

if rpc.is_available():
    from torch.distributed.rpc import RRef  # pylint: disable=ungrouped-imports,unused-import

    __all__.extend(['RRef'])
else:
    RRef = None  # type: ignore[misc,assignment] # pylint: disable=invalid-name
