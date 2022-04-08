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

from typing import NamedTuple, Tuple, Callable

import typing_extensions

from TorchOpt._src import pytypes

OptState = pytypes.TensorTree  # States are arbitrary nests of `torch.Tensor`.
# Parameters are arbitrary nests of `torch.Tensor`.
Params = pytypes.TensorTree
Updates = Params  # Gradient updates are of the same type as parameters.

Schedule = Callable[[pytypes.Numeric], pytypes.Numeric]


class EmptyState(NamedTuple):
    """An empty state for the simplest stateless transformations."""


class TransformInitFn(typing_extensions.Protocol):
    """A callable type for the `init` step of a `GradientTransformation`.

  The `init` step takes a tree of `params` and uses these to construct an
  arbitrary structured initial `state` for the gradient transformation. This
  may hold statistics of the past updates or any other non static information.
  """

    def __call__(self, params: Params) -> OptState:
        """The `init` function.

    Args:
      params: The initial value of the parameters.

    Returns:
      The initial state of the gradient transformation.
    """
        ...


class TransformUpdateFn(typing_extensions.Protocol):
    """A callable type for the `update` step of a `GradientTransformation`.

  The `update` step takes a tree of candidate parameter `updates` (e.g. their
  gradient with respect to some loss), an arbitrary structured `state`, and the
  current `params` of the model being optimised. The `params` argument is
  optional, it must however be provided when using transformations that require
  access to the current values of the parameters.
  """

    def __call__(
            self,
            updates: Updates,
            state: OptState,
            inplace: bool = True
    ) -> Tuple[Updates, OptState]:
        """The `update` function.

    Args:
      updates: A tree of candidate updates.
      state: The state of the gradient transformation.
      inplace: (Optionally) if true, modify updates and state using inplace operations.

    Returns:
      The transformed updates, and the updated state.
    """
        ...


class GradientTransformation(NamedTuple):
    """A pair of pure functions implementing a gradient transformation.

  TorchOpt optimizers are all implemented as _gradient transformations_ like
  Optax. A gradient transformation is defined to be a pair of pure functions,
  which are combined together in a `NamedTuple` so that they can be referred
  to by name.

  Since gradient transformations do not contain any internal state, all stateful
  optimizer properties (such as the current step count when using optimizer
  scheduels, or momemtum values) are passed through gradient transformations by
  using the optimizer _state_ pytree. Each time a gradient transformation is
  applied, the state is computed and returned, ready to be passed to the next
  call to the gradient transformation.

  Attributes:
    init: A pure function which, when called with an example instance of the
      parameters whose gradients will be transformed, returns a pytree
      containing the initial value for the optimizer state.
    update: A pure function which takes as input a pytree of updates (with the
      same tree structure as the original params pytree passed to init), the
      previous optimizer state (which may have been initialized using the init
      function), and optionally the inplace flag. The update function then
      returns the computed gradient updates, and a updates optimizer state.
      If the inplace flag is true, the output results are the same instance as
      the input.
  """
    init: TransformInitFn
    update: TransformUpdateFn


def identity() -> GradientTransformation:
    """Stateless identity transformation that leaves input gradients untouched.

  This function passes through the *gradient updates* unchanged.

  Returns:
    An (init_fn, update_fn) tuple.
  """

    def init_fn(_):
        return EmptyState()

    def update_fn(updates, state, inplace=False):
        return updates, state

    return GradientTransformation(init_fn, update_fn)
