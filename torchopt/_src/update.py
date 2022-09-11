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
# https://github.com/deepmind/optax/blob/master/optax/_src/update.py
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

from torchopt._src import base  # pylint: disable=unused-import
from torchopt._src.utils import pytree


def apply_updates(
    params: 'base.Params', updates: 'base.Updates', *, inplace: bool = True
) -> 'base.Params':
    """Applies an update to the corresponding parameters.

    This is a utility functions that applies an update to a set of parameters, and then returns the
    updated parameters to the caller. As an example, the update may be a gradient transformed by a
    sequence of :class:`GradientTransformations`. This function is exposed for convenience, but it
    just adds updates and parameters; you may also apply updates to parameters manually, using
    :func:`tree_map` (e.g. if you want to manipulate updates in custom ways before applying them).

    Args:
        params: A tree of parameters.
        updates:
            A tree of updates, the tree structure and the shape of the leaf nodes must match that
            of ``params``.
        inplace: If :data:`True`, will update params in a inplace manner.

    Returns:
        Updated parameters, with same structure, shape and type as ``params``.
    """
    if inplace:

        def f(p, u):
            if u is not None:
                p.data.add_(u)
            return p

    else:

        def f(p, u):
            return p.add(u) if u is not None else p

    return pytree.tree_map(f, params, updates)
