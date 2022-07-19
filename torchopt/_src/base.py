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
# This file is modified from:
# https://github.com/deepmind/optax/blob/master/optax/_src/base.py
# ==============================================================================
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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

from abc import abstractmethod
from typing import Callable, NamedTuple, Tuple

from typing_extensions import Protocol

from torchopt._src.typing import Numeric, TensorTree


OptState = TensorTree  # States are arbitrary nests of `torch.Tensor`.
# Parameters are arbitrary nests of `torch.Tensor`.
Params = TensorTree
Updates = Params  # Gradient updates are of the same type as parameters.

Schedule = Callable[[Numeric], Numeric]


class EmptyState(NamedTuple):
    """An empty state for the simplest stateless transformations."""


class TransformInitFn(Protocol):  # pylint: disable=too-few-public-methods
    """A callable type for the :func:`init` step of a :class:`GradientTransformation`.

    The :func:`init` step takes a tree of ``params`` and uses these to construct an arbitrary
    structured initial ``state`` for the gradient transformation. This may hold statistics of the
    past updates or any other non static information.
    """

    @abstractmethod
    def __call__(self, params: Params) -> OptState:
        """The `init` function.

        Args:
            params:
                The initial value of the parameters.

        Returns:
            The initial state of the gradient transformation.
        """


class TransformUpdateFn(Protocol):  # pylint: disable=too-few-public-methods
    """A callable type for the :func:`update` step of a :class:`GradientTransformation`.

    The :func:`update` step takes a tree of candidate parameter ``updates`` (e.g. their gradient
    with respect to some loss), an arbitrary structured ``state``, and the current ``params`` of the
    model being optimized. The ``params`` argument is optional, it must however be provided when
    using transformations that require access to the current values of the parameters.
    """

    @abstractmethod
    def __call__(
        self, updates: Updates, state: OptState, inplace: bool = True
    ) -> Tuple[Updates, OptState]:
        """The `update` function.

        Args:
            updates: A tree of candidate updates.
            state: The state of the gradient transformation.
            inplace: (optional)
                If :data:`True`, modify updates and state using inplace operations.

        Returns:
            The transformed ``updates``, and the updated ``state``.
        """


class GradientTransformation(NamedTuple):
    """A pair of pure functions implementing a gradient transformation.

    TorchOpt optimizers are all implemented as *gradient transformations* like Optax. A gradient
    transformation is defined to be a pair of pure functions, which are combined together in a
    :class:`NamedTuple` so that they can be referred to by name.

    Since gradient transformations do not contain any internal state, all stateful optimizer
    properties (such as the current step count when using optimizer schedules, or momentum values)
    are passed through gradient transformations by using the optimizer *state* ``pytree``. Each time
    a gradient transformation is applied, the state is computed and returned, ready to be passed to
    the next call to the gradient transformation.

    Attributes:
        init:
            A pure function which, when called with an example instance of the parameters whose
            gradients will be transformed, returns a ``pytree`` containing the initial value for the
            optimizer state.
        update:
            A pure function which takes as input a pytree of updates (with the same tree structure
            as the original params ``pytree`` passed to :attr:`init`), the previous optimizer state
            (which may have been initialized using the :attr:`init` function), and optionally the
            ``inplace`` flag. The :attr:`update` function then returns the computed gradient
            updates, and a updates optimizer state. If the ``inplace`` flag is :data:`True`, the
            output results are the same instance as the input.
    """

    init: TransformInitFn
    update: TransformUpdateFn


def identity() -> GradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

    This function passes through the *gradient updates* unchanged.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """

    def init_fn(_):
        return EmptyState()

    def update_fn(updates, state, inplace=False):  # pylint: disable=unused-argument
        return updates, state

    return GradientTransformation(init_fn, update_fn)
