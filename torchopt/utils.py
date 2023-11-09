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
"""Utilities for TorchOpt."""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Sequence, cast, overload
from typing_extensions import TypeAlias  # Python 3.10+

import torch
import torch.nn as nn

from torchopt import pytree
from torchopt.typing import Device, ModuleTensorContainers, OptState, TensorContainer, TensorTree


if TYPE_CHECKING:
    from torchopt.optim.meta.base import MetaOptimizer


__all__ = [
    'ModuleState',
    'stop_gradient',
    'extract_state_dict',
    'recover_state_dict',
    'module_clone',
    'module_detach_',
]


class ModuleState(NamedTuple):
    """Container for module state."""

    params: tuple[TensorContainer, ...]
    buffers: tuple[TensorContainer, ...]
    visual_contents: dict | None = None
    detach_buffers: bool = False


CopyMode: TypeAlias = Literal['reference', 'copy', 'deepcopy', 'ref', 'clone', 'deepclone']


def stop_gradient(target: ModuleState | nn.Module | MetaOptimizer | TensorTree) -> None:
    """Stop the gradient for the input object.

    Since a tensor use ``grad_fn`` to connect itself with the previous computation graph, the
    backpropagated gradient will flow over the tensor and continue flow to the tensors that is
    connected by ``grad_fn``. Some algorithms requires manually detaching tensors from the
    computation graph.

    Note that the :func:`stop_gradient` operation is in-place.

    Args:
        target (ModuleState, nn.Module, MetaOptimizer, or tree of Tensor): The target that to be
            detached from the computation graph, it could be a :class:`nn.Module`,
            :class:`torchopt.MetaOptimizer`, state of the :class:`torchopt.MetaOptimizer`, or just
            a plain list of tensors.
    """
    # pylint: disable-next=import-outside-toplevel
    from torchopt.optim.meta.base import MetaOptimizer

    def fn_(obj: Any) -> None:
        if isinstance(obj, torch.Tensor):
            requires_grad = obj.requires_grad
            obj.detach_().requires_grad_(requires_grad)

    if isinstance(target, ModuleState):
        true_target = cast(TensorTree, (target.params, target.buffers))
    elif isinstance(target, nn.Module):
        true_target = cast(TensorTree, tuple(target.parameters()))
    elif isinstance(target, MetaOptimizer):
        true_target = cast(TensorTree, target.state_dict())
    else:
        true_target = cast(TensorTree, target)  # tree of tensors

    pytree.tree_map_(fn_, true_target)


@overload
def extract_state_dict(  # pylint: disable=too-many-arguments
    target: nn.Module,
    *,
    by: CopyMode = 'reference',
    device: Device | None = None,
    with_buffers: bool = True,
    detach_buffers: bool = False,
    enable_visual: bool = False,
    visual_prefix: str = '',
) -> ModuleState:  # pragma: no cover
    ...


@overload
def extract_state_dict(
    target: MetaOptimizer,
    *,
    by: CopyMode = 'reference',
    device: Device | None = None,
) -> tuple[OptState, ...]:  # pragma: no cover
    ...


# pylint: disable-next=too-many-arguments,too-many-branches,too-many-locals
def extract_state_dict(
    target: nn.Module | MetaOptimizer,
    *,
    by: CopyMode = 'reference',
    device: Device | None = None,
    with_buffers: bool = True,
    detach_buffers: bool = False,
    enable_visual: bool = False,
    visual_prefix: str = '',
) -> ModuleState | tuple[OptState, ...]:
    """Extract target state.

    Since a tensor use ``grad_fn`` to connect itself with the previous computation graph, the
    backpropagated gradient will flow over the tensor and continue flow to the tensors that is
    connected by ``grad_fn``. Some algorithms requires manually detaching tensors from the
    computation graph.

    Note that the extracted state is a reference, which means any in-place operator will affect the
    target that the state is extracted from.

    Args:
        target (nn.Module or MetaOptimizer): It could be a :class:`nn.Module` or
            :class:`torchopt.MetaOptimizer`.
        by (str, optional): The extract policy of tensors in the target. (default: :const:`'reference'`)
            - :const:`'reference'`: The extracted tensors will be references to the original
            tensors.
            - :const:`'copy'`: The extracted tensors will be clones of the original tensors. This
            makes the copied tensors have ``grad_fn`` to be a ``<CloneBackward>`` function points
            to the original tensors.
            - :const:`'deepcopy'`: The extracted tensors will be deep-copied from the original
            tensors. The deep-copied tensors will detach from the original computation graph.
        device (Device or None, optional): If specified, move the extracted state to the specified
            device. (default: :const:`None`)
        with_buffers (bool, optional): Extract buffer together with parameters, this argument is
            only used if the input target is :class:`nn.Module`. (default: :const:`True`)
        detach_buffers (bool, optional): Whether to detach the reference to the buffers, this
            argument is only used if the input target is :class:`nn.Module` and ``by='reference'``.
            (default: :const:`False`)
        enable_visual (bool, optional): Add additional annotations, which could be used in
            computation graph visualization. Currently, this flag only has effect on
            :class:`nn.Module` but we will support :class:`torchopt.MetaOptimizer` later.
            (default: :const:`False`)
        visual_prefix (str, optional): Prefix for the visualization annotations.
            (default: :const:`''`)

    Returns:
        State extracted of the input object.
    """
    assert by in ('reference', 'copy', 'deepcopy', 'ref', 'clone', 'deepclone')
    by = by.replace('clone', 'copy')
    by = 'reference' if by == 'ref' else by

    # pylint: disable=import-outside-toplevel
    from torchopt.optim.meta.base import MetaOptimizer

    if device is not None:
        target_device = torch.device(device)

        def reference(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=target_device)

        def clone(t: torch.Tensor) -> torch.Tensor:
            return t.clone().to(device=target_device)

        def clone_detach_(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, nn.Parameter):
                return nn.Parameter(
                    t.clone().to(device=target_device).detach_(),
                    requires_grad=t.requires_grad,
                )
            return t.clone().to(device=target_device).detach_().requires_grad_(t.requires_grad)

    else:

        def reference(t: torch.Tensor) -> torch.Tensor:
            return t

        def clone(t: torch.Tensor) -> torch.Tensor:
            return t.clone()

        def clone_detach_(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, nn.Parameter):
                return nn.Parameter(t.clone().detach_(), requires_grad=t.requires_grad)
            return t.clone().detach_().requires_grad_(t.requires_grad)

    if by == 'reference':
        replicate = reference
    elif by == 'copy':
        replicate = clone
    else:
        replicate = clone_detach_

    if isinstance(target, nn.Module):  # pylint: disable=no-else-return
        if enable_visual:
            visual_contents = {}

            for k, v in target.named_parameters():  # pylint: disable=invalid-name
                if v.grad_fn is not None:
                    visual_contents.update({v.grad_fn: (visual_prefix + k, v)})
                else:
                    visual_contents.update({v: visual_prefix + k})  # type: ignore[dict-item]
        else:
            visual_contents = None

        params: list[TensorContainer] = []
        buffers: list[TensorContainer] = []
        memo: set[nn.Module] = set()

        def update_params(container: TensorContainer) -> None:
            if len(container) > 0:
                params.append(
                    type(container)(
                        (k, replicate(v))
                        for k, v in container.items()
                        if isinstance(v, torch.Tensor)
                    ),
                )

        def update_buffers(container: TensorContainer) -> None:
            if len(container) > 0:
                fn = clone_detach_ if detach_buffers else replicate
                buffers.append(
                    type(container)(
                        (k, fn(v)) for k, v in container.items() if isinstance(v, torch.Tensor)
                    ),
                )

        # pylint: disable=protected-access
        update_params(target._parameters)  # type: ignore[arg-type]
        if with_buffers:
            update_buffers(target._buffers)
        memo.add(target)
        for submodule in target.modules():
            if submodule in memo:
                continue
            update_params(submodule._parameters)  # type: ignore[arg-type]
            if with_buffers:
                update_buffers(submodule._buffers)
            memo.add(submodule)

        return ModuleState(
            params=tuple(params),
            buffers=tuple(buffers),
            visual_contents=visual_contents,
            detach_buffers=detach_buffers,
        )

    if isinstance(target, MetaOptimizer):
        state = target.state_dict()

        def get_variable(t: torch.Tensor | None) -> torch.Tensor | None:
            if isinstance(t, torch.Tensor):
                return replicate(t)
            return t

        return pytree.tree_map(get_variable, state)  # type: ignore[arg-type,return-value]

    raise RuntimeError(f'Unexpected class of {target}')


def extract_module_containers(
    module: nn.Module,
    with_buffers: bool = True,
) -> tuple[ModuleTensorContainers, ModuleTensorContainers]:
    """Extract the references to the containers of parameters and buffers from a module."""
    if isinstance(module, nn.Module):
        params: list[TensorContainer] = []
        buffers: list[TensorContainer] = []
        memo: set[nn.Module] = set()

        def update_container(container: list[TensorContainer], items: TensorContainer) -> None:
            if len(items) > 0:
                container.append(items)  # we need references to original dictionaries

        # pylint: disable=protected-access
        update_container(params, module._parameters)  # type: ignore[arg-type]
        if with_buffers:
            update_container(buffers, module._buffers)
        memo.add(module)
        for submodule in module.modules():
            if submodule in memo:
                continue
            update_container(params, submodule._parameters)  # type: ignore[arg-type]
            if with_buffers:
                update_container(buffers, submodule._buffers)
            memo.add(submodule)
        return tuple(params), tuple(buffers)

    raise RuntimeError(f'Unexpected class of {module}')


def recover_state_dict(
    target: nn.Module | MetaOptimizer,
    state: ModuleState | Sequence[OptState],
) -> None:
    """Recover state.

    This function is compatible for the ``extract_state``.

    Note that the recovering process is not in-place, so the tensors of the object will not be
    modified.

    Args:
        target (nn.Module or MetaOptimizer): Target that need to recover.
        state (ModuleState or sequence of tree of Tensor): The recovering state.
    """
    # pylint: disable-next=import-outside-toplevel
    from torchopt.optim.meta.base import MetaOptimizer

    if isinstance(target, nn.Module):
        params, buffers, *_ = state = cast(ModuleState, state)
        params_containers, buffers_containers = extract_module_containers(target, with_buffers=True)

        if state.detach_buffers:

            def clone_detach_(t: torch.Tensor) -> torch.Tensor:
                if isinstance(t, nn.Parameter):
                    return nn.Parameter(t.clone().detach_(), requires_grad=t.requires_grad)
                return t.clone().detach_().requires_grad_(t.requires_grad)

            buffers = pytree.tree_map(clone_detach_, buffers)  # type: ignore[assignment,arg-type]

        for tgt, src in itertools.chain(
            zip(params_containers, params),
            zip(buffers_containers, buffers),
        ):
            tgt.update(src)
    elif isinstance(target, MetaOptimizer):
        state = cast(Sequence[OptState], state)
        target.load_state_dict(state)
    else:
        raise RuntimeError(f'Unexpected class of {target}')


@overload
def module_clone(
    target: nn.Module,
    *,
    by: CopyMode = 'reference',
    detach_buffers: bool = False,
    device: Device | None = None,
) -> nn.Module:  # pragma: no cover
    ...


@overload
def module_clone(
    target: MetaOptimizer,
    *,
    by: CopyMode = 'reference',
    detach_buffers: bool = False,
    device: Device | None = None,
) -> MetaOptimizer:  # pragma: no cover
    ...


@overload
def module_clone(
    target: TensorTree,
    *,
    by: CopyMode = 'reference',
    detach_buffers: bool = False,
    device: Device | None = None,
) -> TensorTree:  # pragma: no cover
    ...


# pylint: disable-next=too-many-locals
def module_clone(
    target: nn.Module | MetaOptimizer | TensorTree,
    *,
    by: CopyMode = 'reference',
    detach_buffers: bool = False,
    device: Device | None = None,
) -> nn.Module | MetaOptimizer | TensorTree:
    """Clone a module.

    Args:
        target (nn.Module, MetaOptimizer, or tree of Tensor): The target to be cloned.
        by (str, optional): The extract policy of tensors in the target. (default: :const:`'reference'`)
            - :const:`'reference'`: The extracted tensors will be references to the original
            tensors.
            - :const:`'copy'`: The extracted tensors will be clones of the original tensors. This
            makes the copied tensors have ``grad_fn`` to be a ``<CloneBackward>`` function points
            to the original tensors.
            - :const:`'deepcopy'`: The extracted tensors will be deep-copied from the original
            tensors. The deep-copied tensors will detach from the original computation graph.
        detach_buffers (bool, optional): Whether to detach the reference to the buffers, this
            argument is only used if the input target is :class:`nn.Module` and ``by='reference'``.
            (default: :const:`False`)
        device (Device or None, optional): If specified, move the cloned module to the specified
            device. (default: :const:`None`)

    Returns:
        The cloned module.
    """
    assert by in ('reference', 'copy', 'deepcopy', 'ref', 'clone', 'deepclone')
    by = by.replace('clone', 'copy')
    by = 'reference' if by == 'ref' else by
    if device is not None:
        device = torch.device(device)

    # pylint: disable-next=import-outside-toplevel
    from torchopt.optim.meta.base import MetaOptimizer

    if isinstance(target, (nn.Module, MetaOptimizer)):
        if isinstance(target, nn.Module):
            containers = cast(TensorTree, extract_module_containers(target, with_buffers=True))
        else:
            containers = cast(TensorTree, target.state_dict())
        tensors = pytree.tree_leaves(containers)
        memo = {id(t): t for t in tensors}
        cloned = copy.deepcopy(target, memo=memo)
        state = extract_state_dict(  # type: ignore[call-overload]
            target,
            by=by,
            with_buffers=True,
            detach_buffers=detach_buffers,
            device=device,
        )
        recover_state_dict(cloned, state)
        return cloned

    # Tree of tensors
    if device is not None:
        target_device = torch.device(device)

        def reference(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=target_device)

        def clone(t: torch.Tensor) -> torch.Tensor:
            return t.clone().to(device=target_device)

        def clone_detach_(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, nn.Parameter):
                return nn.Parameter(
                    t.clone().to(device=target_device).detach_(),
                    requires_grad=t.requires_grad,
                )
            return t.clone().to(device=target_device).detach_().requires_grad_(t.requires_grad)

    else:

        def reference(t: torch.Tensor) -> torch.Tensor:
            return t

        def clone(t: torch.Tensor) -> torch.Tensor:
            return t.clone()

        def clone_detach_(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, nn.Parameter):
                return nn.Parameter(t.clone().detach_(), requires_grad=t.requires_grad)
            return t.clone().detach_().requires_grad_(t.requires_grad)

    if by == 'reference':
        replicate = reference
    elif by == 'copy':
        replicate = clone
    else:
        replicate = clone_detach_

    return pytree.tree_map(replicate, cast(TensorTree, target))


@overload
def module_detach_(target: ModuleState) -> ModuleState:  # pragma: no cover
    ...


@overload
def module_detach_(target: nn.Module) -> nn.Module:  # pragma: no cover
    ...


@overload
def module_detach_(target: MetaOptimizer) -> MetaOptimizer:  # pragma: no cover
    ...


@overload
def module_detach_(target: TensorTree) -> TensorTree:  # pragma: no cover
    ...


def module_detach_(
    target: ModuleState | nn.Module | MetaOptimizer | TensorTree,
) -> ModuleState | nn.Module | MetaOptimizer | TensorTree:
    """Detach a module from the computation graph.

    Args:
        target (ModuleState, nn.Module, MetaOptimizer, or tree of Tensor): The
            target to be detached.

    Returns:
        The detached module.
    """
    stop_gradient(target)
    return target
