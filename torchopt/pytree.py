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
"""The PyTree utilities."""

import optree
import optree.typing as typing  # pylint: disable=unused-import
import torch.distributed.rpc as rpc
from optree import *  # pylint: disable=wildcard-import,unused-wildcard-import

from torchopt.typing import Future, PyTree, RRef, T


__all__ = [*optree.__all__, 'tree_wait']


def tree_wait(future_tree: PyTree[Future[T]]) -> PyTree[T]:
    r"""Convert a tree of :class:`Future`\s to a tree of results."""
    import torch  # pylint: disable=import-outside-toplevel

    futures, treedef = tree_flatten(future_tree)

    results = torch.futures.wait_all(futures)

    return tree_unflatten(treedef, results)


if rpc.is_available():

    def tree_as_rref(tree: PyTree[T]) -> 'PyTree[RRef[T]]':
        r"""Convert a tree of local objects to a tree of :class:`RRef`\s."""
        # pylint: disable-next=import-outside-toplevel,redefined-outer-name,reimported
        from torch.distributed.rpc import RRef

        return tree_map(RRef, tree)

    def tree_to_here(
        rref_tree: 'PyTree[RRef[T]]',
        timeout: float = rpc.api.UNSET_RPC_TIMEOUT,
    ) -> PyTree[T]:
        r"""Convert a tree of :class:`RRef`\s to a tree of local objects."""
        return tree_map(lambda x: x.to_here(timeout=timeout), rref_tree)

    def tree_local_value(rref_tree: 'PyTree[RRef[T]]'):
        r"""Return the local value of a tree of :class:`RRef`\s."""
        return tree_map(lambda x: x.local_value(), rref_tree)

    __all__.extend(['tree_as_rref', 'tree_to_here'])


del optree, rpc, PyTree, T, RRef
