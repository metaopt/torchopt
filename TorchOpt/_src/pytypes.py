from typing import Any, Callable, Iterable, Mapping, Union

from torch import Tensor

Scalar = Union[float, int]
Numeric = Union[Tensor, Scalar]
TensorTree = Union[Tensor, Iterable['TensorTree'], Mapping[Any, 'TensorTree']]

Schedule = Callable[[Numeric], Numeric]
ScalarOrSchedule = Union[float, Schedule]
