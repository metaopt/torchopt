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
"""The base classes for gradient transformation."""

from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, NamedTuple, Protocol
from typing_extensions import Self  # Python 3.11+


if TYPE_CHECKING:
    from torchopt.typing import OptState, Params, Updates


__all__ = [
    'EmptyState',
    'UninitializedState',
    'GradientTransformation',
    'ChainedGradientTransformation',
    'identity',
]


class EmptyState(NamedTuple):
    """An empty state for the simplest stateless transformations."""


class UninitializedState(NamedTuple):
    """A state that is not initialized yet."""


class TransformInitFn(Protocol):  # pylint: disable=too-few-public-methods
    """A callable type for the :func:`init` step of a :class:`GradientTransformation`.

    The :func:`init` step takes a tree of ``params`` and uses these to construct an arbitrary
    structured initial ``state`` for the gradient transformation. This may hold statistics of the
    past updates or any other non static information.
    """

    @abstractmethod
    def __call__(self, params: Params) -> OptState:
        """Initialize the gradient transformation state.

        Args:
            params (tree of Tensor): The initial value of the parameters.

        Returns:
            The initial state of the gradient transformation.
        """


class TransformUpdateFn(Protocol):  # pylint: disable=too-few-public-methods
    """A callable type for the :func:`update` step of a :class:`GradientTransformation`.

    The :func:`update` step takes a tree of candidate parameter ``updates`` (e.g. their gradient
    with respect to some loss), an arbitrary structured ``state``, and the current ``params`` of the
    model being optimized. The ``params`` argument is optional, it must however be provided when
    using transformations that require access to the current values of the parameters. The
    ``inplace`` argument is optional, If :data:`True`, modify updates and state using inplace
    operations.
    """

    @abstractmethod
    def __call__(
        self,
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        """Transform the updates and state.

        Args:
            updates (tree of Tensor): A tree of candidate updates.
            state (tree of Tensor): The state of the gradient transformation.
            params (tree of Tensor or None, optional): The current value of the parameters.
                (default: :data:`None`)
            inplace (bool, optional): If :data:`True`, modify updates and state using inplace
                operations. (default: :data:`True`)

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
            as the original params ``pytree`` passed to ``init``), the previous optimizer state
            (which may have been initialized using the ``init`` function), and optionally the
            ``inplace`` flag. The ``update`` function then returns the computed gradient
            updates, and a updates optimizer state. If the ``inplace`` flag is :data:`True`, the
            output results are the same instance as the input.
    """

    init: TransformInitFn
    update: TransformUpdateFn

    # pylint: disable-next=redefined-builtin
    def chain(self, next: GradientTransformation) -> ChainedGradientTransformation:
        """Chain two gradient transformations together."""
        return ChainedGradientTransformation(self, next)


class ChainedGradientTransformation(GradientTransformation):
    """A chain of gradient transformations.

    This class is a subclass of :class:`GradientTransformation` which allows for chaining of
    gradient transformations.
    """

    transformations: tuple[GradientTransformation, ...]

    def __new__(cls, *transformations: GradientTransformation) -> Self:
        """Create a new chained gradient transformation."""
        transformations = tuple(
            itertools.chain.from_iterable(
                t.transformations
                if isinstance(t, ChainedGradientTransformation)
                else ((t,) if not isinstance(t, IdentityGradientTransformation) else ())
                for t in transformations
            ),
        )

        if len(transformations) == 0:
            transformations = (IdentityGradientTransformation(),)

        init_fns, update_fns = tuple(zip(*transformations))

        def init_fn(params: Params) -> OptState:
            return tuple(fn(params) for fn in init_fns)

        def update_fn(
            updates: Updates,
            state: OptState,
            *,
            params: Params | None = None,
            inplace: bool = True,
        ) -> tuple[Updates, OptState]:
            if len(update_fns) != len(state):
                raise ValueError(
                    'The number of updates and states has to be the same in chain! Make sure you'
                    'have called init first!',
                )
            new_state = []
            for s, fn in zip(state, update_fns):  # pylint: disable=invalid-name
                updates, new_s = fn(updates, s, params=params, inplace=inplace)
                new_state.append(new_s)
            return updates, tuple(new_state)

        instance = super().__new__(cls, init=init_fn, update=update_fn)
        instance.transformations = transformations
        return instance

    def __repr__(self) -> str:
        """Return a string representation of the chained gradient transformation."""
        return '{}(\n    {},\n)'.format(
            self.__class__.__name__,
            ',\n    '.join(repr(t) for t in self.transformations),
        )

    def __eq__(self, other: object) -> bool:
        """Return whether two chained gradient transformations are equal."""
        if isinstance(other, ChainedGradientTransformation):
            return self.transformations == other.transformations
        if isinstance(other, GradientTransformation):
            return self.transformations == (other,)
        return False

    def __hash__(self) -> int:
        """Return the hash of the chained gradient transformation."""
        return hash(self.transformations)

    def __getstate__(self) -> tuple[GradientTransformation, ...]:
        """Return the state of the chained gradient transformation for serialization."""
        return self.transformations

    def __setstate__(self, state: tuple[GradientTransformation, ...]) -> None:
        """Set the state of the chained gradient transformation from serialization."""
        self.transformations = state

    def __reduce__(self) -> tuple[Callable, tuple[tuple[GradientTransformation, ...]]]:
        """Serialize the chained gradient transformation."""
        return ChainedGradientTransformation, (self.transformations,)


class IdentityGradientTransformation(GradientTransformation):
    """A gradient transformation that does nothing."""

    def __new__(cls) -> Self:
        """Create a new gradient transformation that does nothing."""
        return super().__new__(cls, init=cls.init_fn, update=cls.update_fn)

    @staticmethod
    def init_fn(params: Params) -> OptState:  # pylint: disable=unused-argument
        """Return empty state."""
        return EmptyState()

    @staticmethod
    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,  # pylint: disable=unused-argument
        inplace: bool = True,  # pylint: disable=unused-argument
    ) -> tuple[Updates, OptState]:
        """Return updates unchanged."""
        return updates, state


def identity() -> IdentityGradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

    This function passes through the *gradient updates* unchanged.

    Returns:
        An ``(init_fn, update_fn)`` tuple.
    """
    return IdentityGradientTransformation()
