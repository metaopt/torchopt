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
"""Distributed APIs."""

from __future__ import annotations

import functools
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.distributed.rpc as rpc

from torchopt import pytree
from torchopt.distributed.world import get_worker_id, get_world_rank, get_world_size
from torchopt.typing import Future


__all__ = [
    'TensorDimensionPartitioner',
    'dim_partitioner',
    'batch_partitioner',
    'mean_reducer',
    'sum_reducer',
    'remote_async_call',
    'remote_sync_call',
    'parallelize',
    'parallelize_async',
    'parallelize_sync',
]


UNSET_RPC_TIMEOUT = rpc.api.UNSET_RPC_TIMEOUT if rpc.is_available() else -1.0


T = TypeVar('T')
U = TypeVar('U')
Args = Tuple[Any, ...]
KwArgs = Dict[str, Any]
PartitionFunction = Callable[..., Sequence[Tuple[int, Optional[Args], Optional[KwArgs]]]]
Partitioner = Union[int, str, PartitionFunction]


class TensorDimensionPartitioner:
    """Partitioner class that partitions a batch of inputs along a given dimension.

    All tensors in the ``args`` and ``kwargs`` will be partitioned along the dimension ``dim``,
    while the non-tensor values will be broadcasted to partitions.

    Args:
        dim (int): The dimension to partition.
        exclusive (bool, optional): Whether to partition the batch exclusively. (default: :data:`False`)
            If :data:`True`, the batch will be partitioned into ``batch_size`` partitions, where
            ``batch_size`` is the size of the batch along the given dimension. Each batch sample
            will be assigned to a separate RPC call.
            If :data:`False`, the batch will be partitioned into ``min(batch_size, num_workers)``
            partitions, where ``num_workers`` is the number of workers in the world. When
            ``batch_size > num_workers``, there can be multiple batch samples forward in a single
            RPC call.
        keepdim (bool, optional): Whether to keep the partitioned dimension. (default: :data:`True`)
            If :data:`True`, keep the batch dimension. If :data:`False`, use select instead of
            slicing. This functionality should be used with ``exclusive=True``.
        workers (sequence of int or str, or None, optional): The workers to partition the batch to.
            If :data:`None`, the batch will be partitioned to all workers in the world.
            (default: :data:`None`)
    """

    def __init__(
        self,
        dim: int,
        *,
        exclusive: bool = False,
        keepdim: bool = False,
        workers: Sequence[int | str] | None = None,
    ) -> None:
        """Initialize the partitioner instance."""
        if not keepdim and not exclusive:
            raise ValueError('keepdim=False should be used with exclusive=True.')

        self.dim = dim
        self.exclusive = exclusive
        self.keepdim = keepdim
        self.workers = workers

    # pylint: disable-next=too-many-branches,too-many-locals
    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[int, Args | None, KwArgs | None]]:
        """Partition the batch of inputs along the given dimension."""
        if self.workers is None:
            workers = list(range(get_world_size()))
        else:
            workers = list(map(get_worker_id, self.workers))
        num_workers = len(workers)

        args_tree = (args, kwargs)
        flat_args: list[Any]
        flat_args, treespec = pytree.tree_flatten(args_tree)  # type: ignore[arg-type]

        batch_size = None
        for arg in flat_args:
            if isinstance(arg, torch.Tensor):
                if batch_size is None:
                    batch_size = arg.shape[self.dim]
                elif batch_size != arg.shape[self.dim]:  # type: ignore[unreachable]
                    raise ValueError(
                        f'Batch size mismatch on dim={self.dim}. '
                        f'Expected {batch_size}, got {arg.shape[self.dim]} (shape: {arg.shape}).',
                    )

        if batch_size is None:
            return [(get_world_rank(), args, kwargs.copy())]

        dim_slices: list[int | slice]
        batch_slices: list[tuple[int | slice | Ellipsis.__class__, ...]]  # type: ignore[name-defined]
        if self.exclusive:
            num_replicas = batch_size
            if self.keepdim:
                dim_slices = [slice(i, i + 1) for i in range(num_replicas)]
            else:
                dim_slices = list(range(num_replicas))
        else:
            if batch_size <= num_workers:
                num_replicas = batch_size
                dim_slices = [slice(i, i + 1) for i in range(batch_size)]  # keepdim=True
            else:
                num_replicas = num_workers
                local_size = batch_size // num_workers
                local_batch_indices = [i * local_size for i in range(num_workers)] + [batch_size]
                dim_slices = [
                    slice(local_batch_indices[i], local_batch_indices[i + 1])
                    for i in range(num_workers)
                ]

        if self.dim >= 0:
            batch_slices = [
                (slice(None, None),) * self.dim + (dim_slice,) for dim_slice in dim_slices
            ]
        elif self.dim < 0:
            batch_slices = [
                (
                    ...,
                    dim_slice,
                )
                + (slice(None, None),) * (-self.dim - 1)
                for dim_slice in dim_slices
            ]

        flat_args_replicas: list[list[Any]] = [[] for _ in range(num_replicas)]
        for arg in flat_args:
            if isinstance(arg, torch.Tensor):
                for i, batch_slice in enumerate(batch_slices):
                    flat_args_replicas[i].append(arg[batch_slice])
            else:
                for i in range(num_replicas):
                    flat_args_replicas[i].append(arg)

        args_replicas: list[tuple[Args, KwArgs]] = [
            pytree.tree_unflatten(treespec, args_replica)  # type: ignore[misc]
            for args_replica in flat_args_replicas
        ]

        return [
            (workers[i % num_workers], worker_args, worker_kwargs)
            for i, (worker_args, worker_kwargs) in enumerate(args_replicas)
        ]

    def __reduce__(
        self,
    ) -> tuple[
        Callable[..., TensorDimensionPartitioner],
        tuple[int],
        dict[str, bool | Sequence[int | str] | None],
    ]:
        """Return a tuple that allows the partitioner to be pickled."""
        return (
            TensorDimensionPartitioner,
            (self.dim,),
            {'exclusive': self.exclusive, 'keepdim': self.keepdim, 'workers': self.workers},
        )


def dim_partitioner(
    dim: int = 0,
    *,
    exclusive: bool = False,
    keepdim: bool = True,
    workers: Sequence[int | str] | None = None,
) -> PartitionFunction:
    """Partition a batch of inputs along a given dimension.

    All tensors in the ``args`` and ``kwargs`` will be partitioned along the dimension ``dim``,
    while the non-tensor values will be broadcasted to partitions.

    Args:
        dim (int, optional): The dimension to partition. (default: :const:`0`)
        exclusive (bool, optional): Whether to partition the batch exclusively. (default: :data:`False`)
            If :data:`True`, the batch will be partitioned into ``batch_size`` partitions, where
            ``batch_size`` is the size of the batch along the given dimension. Each batch sample
            will be assigned to a separate RPC call.
            If :data:`False`, the batch will be partitioned into ``min(batch_size, num_workers)``
            partitions, where ``num_workers`` is the number of workers in the world. When
            ``batch_size > num_workers``, there can be multiple batch samples forward in a single
            RPC call.
        keepdim (bool, optional): Whether to keep the partitioned dimension. (default: :data:`False`)
            If :data:`True`, keep the batch dimension. If :data:`False`, use select instead of
            slicing. This functionality should be used with ``exclusive=True``.
        workers (sequence of int or str, or None, optional): The workers to partition the batch to.
            If :data:`None`, the batch will be partitioned to all workers in the world.
            (default: :data:`None`)

    Returns:
        A partition function.
    """
    return TensorDimensionPartitioner(dim, exclusive=exclusive, keepdim=keepdim, workers=workers)


batch_partitioner: PartitionFunction = dim_partitioner(dim=0, keepdim=True, exclusive=False)
"""Partitioner for batch dimension. Divide and replicates the arguments to all workers along the first dimension.

The batch will be partitioned into ``min(batch_size, num_workers)`` partitions, where
``num_workers`` is the number of workers in the world.
When ``batch_size > num_workers``, there can be multiple batch samples forward in a single RPC call.

All tensors in the ``args`` and ``kwargs`` will be partitioned along the dimension ``dim``,
while the non-tensor values will be broadcasted to partitions.
"""
exclusive_batch_partitioner: PartitionFunction = dim_partitioner(dim=0, keepdim=True, exclusive=True)  # fmt: skip
"""Partitioner for batch dimension. Divide and replicates the arguments to all workers along the first dimension.

Each batch sample will be assigned to a separate RPC call.

All tensors in the ``args`` and ``kwargs`` will be partitioned along the dimension ``dim``,
while the non-tensor values will be broadcasted to partitions.
"""


def mean_reducer(results: Iterable[torch.Tensor]) -> torch.Tensor:
    """Reduce the results by averaging them."""
    return torch.mean(torch.stack(tuple(results), dim=0), dim=0)


def sum_reducer(results: Iterable[torch.Tensor]) -> torch.Tensor:
    """Reduce the results by summing them."""
    return torch.sum(torch.stack(tuple(results), dim=0), dim=0)


# pylint: disable-next=too-many-arguments
def remote_async_call(
    func: Callable[..., T],
    *,
    args: Args | None = None,
    kwargs: KwArgs | None = None,
    partitioner: Partitioner | None = None,
    reducer: Callable[[Iterable[T]], U] | None = None,
    timeout: float | None = UNSET_RPC_TIMEOUT,
) -> Future[list[T]] | Future[U]:
    """Asynchronously do an RPC on remote workers and return the a :class:`torch.Future` instance at the current worker.

    Args:
        func (callable): The function to call.
        args (tuple of object or None, optional): The arguments to pass to the function.
            (default: :data:`None`)
        kwargs (dict[str, object] or None, optional): The keyword arguments to pass to the function.
            (default: :data:`None`)
        partitioner (int, str, or callable, optional): A partitioner that partitions the arguments
            to multiple workers. (default: :func:`batch_partitioner`)
        reducer (callable or None, optional): A reducer that reduces the results from multiple
            workers. If :data:`None`, do not reduce the results. (default: :data:`None`)
        timeout (float, optional): The timeout for the RPC call.
            (default: :data:`rpc.api.UNSET_RPC_TIMEOUT`)

    Returns:
        A :class:`torch.Future` instance for the result. The result is at the current worker.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if partitioner is None:
        partitioner = batch_partitioner
    if isinstance(partitioner, (int, str)):
        partitions = [(get_worker_id(id=partitioner), args, kwargs)]
    elif callable(partitioner):
        partitions = partitioner(*args, **kwargs)  # type: ignore[assignment]
    else:
        raise ValueError(f'Invalid partitioner: {partitioner!r}.')

    futures = []
    for rank, worker_args, worker_kwargs in partitions:
        fut = rpc.rpc_async(rank, func, args=worker_args, kwargs=worker_kwargs, timeout=timeout)
        futures.append(fut)

    future = cast(
        Future[List[T]],
        torch.futures.collect_all(futures).then(lambda fut: [f.wait() for f in fut.wait()]),
    )
    if reducer is not None:
        return cast(
            Future[U],
            future.then(lambda fut: cast(Callable[[Iterable[T]], U], reducer)(fut.wait())),
        )
    return future


# pylint: disable-next=too-many-arguments
def remote_sync_call(
    func: Callable[..., T],
    *,
    args: Args | None = None,
    kwargs: KwArgs | None = None,
    partitioner: Partitioner | None = None,
    reducer: Callable[[Iterable[T]], U] | None = None,
    timeout: float | None = UNSET_RPC_TIMEOUT,
) -> list[T] | U:
    """Do an RPC synchronously on remote workers and return the result to the current worker.

    Args:
        func (callable): The function to call.
        args (tuple of object or None, optional): The arguments to pass to the function.
            (default: :data:`None`)
        kwargs (dict[str, object] or None, optional): The keyword arguments to pass to the function.
            (default: :data:`None`)
        partitioner (int, str, or callable, optional): A partitioner that partitions the arguments
            to multiple workers. (default: :func:`batch_partitioner`)
        reducer (callable or None, optional): A reducer that reduces the results from multiple
            workers. If :data:`None`, do not reduce the results. (default: :data:`None`)
        timeout (float, optional): The timeout for the RPC call.
            (default: :data:`rpc.api.UNSET_RPC_TIMEOUT`)

    Returns:
        The result of the RPC call. The result is at the current worker.
    """
    return remote_async_call(
        func,
        args=args,
        kwargs=kwargs,
        partitioner=partitioner,
        timeout=timeout,
        reducer=reducer,
    ).wait()


def parallelize_async(
    partitioner: Partitioner | None = None,
    reducer: Callable[[Iterable[T]], U] | None = None,
    timeout: float | None = UNSET_RPC_TIMEOUT,
) -> Callable[[Callable[..., T]], Callable[..., Future[list[T]] | Future[U]]]:
    """Return a decorator for parallelizing a function.

    This decorator can be used to parallelize a function call across multiple workers. The
    function will be called asynchronously on remote workers. The decorated function will
    return a :class:`torch.Future` instance of the result.

    Args:
        partitioner (int, str, or callable, optional): A partitioner that partitions the arguments
            to multiple workers. (default: :func:`batch_partitioner`)
        reducer (callable or None, optional): A reducer that reduces the results from multiple
            workers. If :data:`None`, do not reduce the results. (default: :data:`None`)
        timeout (float, optional): The timeout for the RPC call.
            (default: :data:`rpc.api.UNSET_RPC_TIMEOUT`)

    Returns:
        The decorator function.
    """
    if partitioner is None:
        partitioner = batch_partitioner
        if reducer is None:
            reducer = mean_reducer  # type: ignore[assignment]

    def wrapper(func: Callable[..., T]) -> Callable[..., Future[list[T]] | Future[U]]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Future[list[T]] | Future[U]:
            return remote_async_call(
                func,
                args=args,
                kwargs=kwargs,
                partitioner=partitioner,
                reducer=reducer,
                timeout=timeout,
            )

        suffix = '__parallelize_async_unwrapped__'
        module_name = func.__module__
        try:
            name = func.__qualname__
        except AttributeError:
            name = func.__name__
        else:
            func.__qualname__ = f'{func.__qualname__}{suffix}'
        func.__name__ = f'{func.__name__}{suffix}'
        __import__(module_name, level=0)
        module = sys.modules[module_name]
        setattr(module, f'{name}{suffix}', func)

        return wrapped

    return wrapper


def parallelize(
    partitioner: Partitioner | None = None,
    reducer: Callable[[Iterable[T]], U] | None = None,
    timeout: float | None = UNSET_RPC_TIMEOUT,
) -> Callable[[Callable[..., T]], Callable[..., list[T] | U]]:
    """Return a decorator for parallelizing a function.

    This decorator can be used to parallelize a function call across multiple workers.

    Args:
        partitioner (int, str, or callable, optional): A partitioner that partitions the arguments
            to multiple workers. (default: :func:`batch_partitioner`)
        reducer (callable or None, optional): A reducer that reduces the results from multiple
            workers. If :data:`None`, do not reduce the results. (default: :data:`None`)
        timeout (float, optional): The timeout for the RPC call.
            (default: :data:`rpc.api.UNSET_RPC_TIMEOUT`)

    Returns:
        The decorator function.
    """
    if partitioner is None:
        partitioner = batch_partitioner
        if reducer is None:
            reducer = mean_reducer  # type: ignore[assignment]

    def wrapper(func: Callable[..., T]) -> Callable[..., list[T] | U]:
        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> list[T] | U:
            return remote_sync_call(
                func,
                args=args,
                kwargs=kwargs,
                partitioner=partitioner,
                reducer=reducer,
                timeout=timeout,
            )

        suffix = '__parallelize_unwrapped__'
        module_name = func.__module__
        try:
            name = func.__qualname__
        except AttributeError:
            name = func.__name__
        else:
            func.__qualname__ = f'{func.__qualname__}{suffix}'
        func.__name__ = f'{func.__name__}{suffix}'
        __import__(module_name, level=0)
        module = sys.modules[module_name]
        setattr(module, f'{name}{suffix}', func)

        return wrapped

    return wrapper


parallelize_sync = parallelize
