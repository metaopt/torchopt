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

from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union
from typing_extensions import TypeAlias  # Python 3.10+

import torch.distributed.rpc as rpc
from optree.typing import PyTree, PyTreeTypeVar
from torch import Tensor
from torch.futures import Future

from torchopt.base import ChainedGradientTransformation, EmptyState, GradientTransformation


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
    'OptionalTensor',
    'ListOfTensors',
    'TupleOfTensors',
    'SequenceOfTensors',
    'TensorOrTensors',
    'TensorTree',
    'ListOfOptionalTensors',
    'TupleOfOptionalTensors',
    'SequenceOfOptionalTensors',
    'OptionalTensorOrOptionalTensors',
    'OptionalTensorTree',
    'Future',
]

T = TypeVar('T')

Scalar: TypeAlias = Union[float, int, bool]
Numeric: TypeAlias = Union[Tensor, Scalar]

Schedule: TypeAlias = Callable[[Numeric], Numeric]
ScalarOrSchedule: TypeAlias = Union[float, Schedule]

OptionalTensor = Optional[Tensor]

ListOfTensors = List[Tensor]
TupleOfTensors = Tuple[Tensor, ...]
SequenceOfTensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, SequenceOfTensors]
TensorTree: TypeAlias = PyTreeTypeVar('TensorTree', Tensor)  # type: ignore[valid-type]

ListOfOptionalTensors = List[OptionalTensor]
TupleOfOptionalTensors = Tuple[OptionalTensor, ...]
SequenceOfOptionalTensors = Sequence[OptionalTensor]
OptionalTensorOrOptionalTensors = Union[OptionalTensor, SequenceOfOptionalTensors]
OptionalTensorTree: TypeAlias = PyTreeTypeVar('OptionalTensorTree', OptionalTensor)  # type: ignore[valid-type]

# Parameters are arbitrary nests of `torch.Tensor`.
Params: TypeAlias = TensorTree
Updates: TypeAlias = Params  # Gradient updates are of the same type as parameters.
OptState: TypeAlias = TensorTree  # States are arbitrary nests of `torch.Tensor`.

if rpc.is_available():
    from torch.distributed.rpc import RRef  # pylint: disable=ungrouped-imports,unused-import

    __all__.extend(['RRef'])
else:
    RRef = None  # type: ignore[misc,assignment] # pylint: disable=invalid-name
