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
"""Utilities for gathering information about the world."""

from __future__ import annotations

import atexit
import functools
import os
from typing import Any, Callable, Iterable, NamedTuple, TypeVar

import torch.distributed.rpc as rpc
from torch.distributed.elastic.multiprocessing.errors import record


__all__ = [
    'get_world_info',
    'get_world_rank',
    'get_rank',
    'get_world_size',
    'get_local_rank',
    'get_local_world_size',
    'get_worker_id',
    'barrier',
    'auto_init_rpc',
    'on_rank',
    'not_on_rank',
    'rank_zero_only',
    'rank_non_zero_only',
]


def default_worker_name_format(
    world_rank: int,
    world_size: int,
    local_rank: int,  # pylint: disable=unused-argument
    local_world_size: int,  # pylint: disable=unused-argument
) -> str:
    """Get the default worker name format."""
    return f'worker{world_rank:0{len(str(world_size))}d}'


F = TypeVar('F', bound=Callable[..., Any])
_WORKER_NAME_FORMAT: Callable[..., str] = default_worker_name_format


class WorldInfo(NamedTuple):
    """Information about the world."""

    world_rank: int
    world_size: int
    local_rank: int
    local_world_size: int

    @property
    def rank(self) -> int:
        """Get the global world rank of the current worker."""
        return self.world_rank

    @property
    def worker_name(self) -> str:
        """Get the name of the current worker."""
        return _WORKER_NAME_FORMAT(
            world_rank=self.world_rank,
            world_size=self.world_size,
            local_rank=self.local_rank,
            local_world_size=self.local_world_size,
        )


def get_world_info() -> WorldInfo:
    """Get the world information."""
    world_info = getattr(get_world_info, 'world_info', None)

    if world_info is None:
        world_rank = int(os.getenv('RANK', '0'))
        world_size = int(os.getenv('WORLD_SIZE', '1'))
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        local_world_size = int(os.getenv('LOCAL_WORLD_SIZE', '1'))
        world_info = WorldInfo(world_rank, world_size, local_rank, local_world_size)
        # pylint: disable=line-too-long
        get_world_info.world_info = get_world_info.WORLD_INFO = world_info  # type: ignore[attr-defined]
        get_world_info.world_rank = get_world_info.WORLD_RANK = world_rank  # type: ignore[attr-defined]
        get_world_info.rank = get_world_info.RANK = world_rank  # type: ignore[attr-defined]
        get_world_info.world_size = get_world_info.WORLD_SIZE = world_size  # type: ignore[attr-defined]
        get_world_info.local_rank = get_world_info.LOCAL_RANK = local_rank  # type: ignore[attr-defined]
        get_world_info.local_world_size = get_world_info.LOCAL_WORLD_SIZE = local_world_size  # type: ignore[attr-defined]
        # pylint: enable=line-too-long

    return world_info


def get_world_rank() -> int:
    """Get the global world rank of the current worker."""
    return get_world_info().world_rank


get_rank = get_world_rank


def get_world_size() -> int:
    """Get the world size."""
    return get_world_info().world_size


def get_local_rank() -> int:
    """Get the local rank of the current worker on the current node."""
    return get_world_info().local_rank


def get_local_world_size() -> int:
    """Get the local world size on the current node."""
    return get_world_info().local_world_size


get_world_info()


# pylint: disable-next=redefined-builtin,invalid-name
def get_worker_id(id: str | int | None = None) -> int:
    """Get the worker id from the given id."""
    if isinstance(id, int):
        return id
    return rpc.get_worker_info(worker_name=id).id


def barrier(worker_names: Iterable[str] | None = None) -> None:
    r"""Synchronize local and remote RPC processes.

    This will block until all local and remote RPC processes specified under worker_names
    reach this method to wait for all outstanding work to complete.

    Args:
        worker_names (iterable of str or None, optional): The set of workers to synchronize.
            If :data:`None`, all workers. (default: :data:`None`)
    """
    worker_names = {} if worker_names is None else set(worker_names)
    rpc.api._barrier(worker_names)  # pylint: disable=protected-access


def auto_init_rpc(
    worker_init_fn: Callable[[], None] | None = None,
    worker_name_format: Callable[..., str] = default_worker_name_format,
    *,
    backend: rpc.BackendType | None = None,
    rpc_backend_options: rpc.RpcBackendOptions | None = None,
) -> Callable[[F], F]:
    """Return a decorator to automatically initialize RPC on the decorated function."""
    global _WORKER_NAME_FORMAT  # pylint: disable=global-statement
    _WORKER_NAME_FORMAT = worker_name_format

    def wrapper(func: F) -> F:
        world_info = get_world_info()

        @record
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            rpc.init_rpc(
                name=world_info.worker_name,
                rank=world_info.rank,
                world_size=world_info.world_size,
                backend=backend,
                rpc_backend_options=rpc_backend_options,
            )
            atexit.register(rpc.shutdown, graceful=True)
            if worker_init_fn is not None:
                barrier()
                worker_init_fn()
            barrier()
            return func(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return wrapper


def __on_ranks(ranks: Iterable[int], inverse: bool = False) -> Callable[[F], F]:
    ranks = frozenset(ranks)

    def wrapper(func: F) -> F:
        world_rank = get_world_info().world_rank

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if inverse:
                if world_rank not in ranks:
                    return func(*args, **kwargs)
            elif world_rank in ranks:
                return func(*args, **kwargs)
            return None

        return wrapped  # type: ignore[return-value]

    return wrapper


def on_rank(*ranks: int) -> Callable[[F], F]:
    """Return a decorator to mark a function to be executed only on given ranks."""
    return __on_ranks(ranks=ranks, inverse=False)


def not_on_rank(*ranks: int) -> Callable[[F], F]:
    """Return a decorator to mark a function to be executed only on non given ranks."""
    return __on_ranks(ranks=ranks, inverse=True)


def rank_all(func: F) -> F:
    """Return a decorator to mark a function to be executed on all ranks."""
    return func


def rank_zero_only(func: F) -> F:
    """Return a decorator to mark a function to be executed only on rank zero."""
    return on_rank(0)(func)


def rank_non_zero_only(func: F) -> F:
    """Return a decorator to mark a function to be executed only on non rank zero."""
    return not_on_rank(0)(func)
