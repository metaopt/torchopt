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
# https://github.com/deepmind/optax/blob/master/optax/_src/alias.py
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

from torchopt._src import base


def chain(*args: base.GradientTransformation) -> base.GradientTransformation:
    """Applies a list of chainable update transformations.

    Given a sequence of chainable transforms, :func:`chain` returns an :func:`init_fn` that
    constructs a ``state`` by concatenating the states of the individual transforms, and returns an
    :func:`update_fn` which chains the update transformations feeding the appropriate state to each.

    Args:
        *args:
            A sequence of chainable ``(init_fn, update_fn)`` tuples.

    Returns:
        A single ``(init_fn, update_fn)`` tuple.
    """
    return base.ChainedGradientTransformation(*args)
