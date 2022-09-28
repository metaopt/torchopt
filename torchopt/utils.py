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
"""Utilities for TorchOpt."""

from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn

from torchopt import pytree
from torchopt.typing import OptState, TensorTree  # pylint: disable=unused-import


try:
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Literal


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

    params: Tuple[Dict[str, torch.Tensor], ...]
    visual_contents: Optional[Dict] = None


def stop_gradient(target: Union['TensorTree', ModuleState, nn.Module, 'MetaOptimizer']) -> None:
    """Stop the gradient for the input object.

    Since a tensor use :attr:`grad_fn` to connect itself with the previous computation graph, the
    back-propagated gradient will flow over the tensor and continue flow to the tensors that is
    connected by :attr:`grad_fn`. Some algorithms requires manually detaching tensors from the
    computation graph.

    Note that the :func:`stop_gradient` operation is in-place.

    Args:
        target: The target that to be detached from the computation graph, it could be a
            :class:`nn.Module`, :class:`torchopt.MetaOptimizer`, state of the
            :class:`torchopt.MetaOptimizer`, or just a plain list of tensors.
        inplace: If :data:`True`, the target will be detached in-place. if :data:`Frue`, this
            function will return a detached copy of the target. The in-place operation is fast and
            memory efficient but may raise back-propagation error.
    """
    # pylint: disable-next=import-outside-toplevel
    from torchopt.optim.meta.base import MetaOptimizer

    def f(obj):
        if isinstance(obj, torch.Tensor):
            obj.detach_().requires_grad_(obj.requires_grad)

    if isinstance(target, ModuleState):
        true_target = cast(TensorTree, target.params)
    elif isinstance(target, nn.Module):
        true_target = cast(TensorTree, tuple(target.parameters()))
    elif isinstance(target, MetaOptimizer):
        true_target = cast(TensorTree, target.state_dict())
    else:
        true_target = cast(TensorTree, target)  # tree of tensors

    pytree.tree_map(f, true_target)


# pylint: disable-next=too-many-branches,too-many-locals
def extract_state_dict(
    target: Union[nn.Module, 'MetaOptimizer'],
    *,
    by: Literal['reference', 'copy', 'deepcopy'] = 'reference',  # type: ignore[name-defined]
    device: Optional[Union[int, str, torch.device]] = None,
    with_buffer: bool = True,
    enable_visual: bool = False,
    visual_prefix: str = '',
) -> Union[ModuleState, Tuple['OptState', ...]]:
    """Extract target state.

    Since a tensor use :attr:`grad_fn` to connect itself with the previous computation graph, the
    back-propagated gradient will flow over the tensor and continue flow to the tensors that is
    connected by :attr:`grad_fn`. Some algorithms requires manually detaching tensors from the
    computation graph.

    Note that the extracted state is a reference, which means any in-place operator will affect the
    target that the state is extracted from.

    Args:
        target: It could be a :class:`nn.Module` or :class:`torchopt.MetaOptimizer`.
        by: The extract policy of tensors in the target.
            - :const:`'reference'`: The extracted tensors will be references to the original
            tensors.
            - :const:`'copy'`: The extracted tensors will be clones of the original tensors. This
            makes the copied tensors have :attr:`grad_fn` to be a ``<CloneBackward>`` function
            points to the original tensors.
            - :const:`'deepcopy'`: The extracted tensors will be deep-copied from the original
            tensors. The deep-copied tensors will detach from the original computation graph.
        device: If specified, move the extracted state to the specified device.
        with_buffer:
            Extract buffer together with parameters, this argument is only used if the input target
            is :class:`nn.Module`.
        enable_visual:
            Add additional annotations, which could be used in computation graph visualization.
            Currently, this flag only has effect on :class:`nn.Module` but we will support
            :class:`torchopt.MetaOptimizer` later.
        visual_prefix: Prefix for the visualization annotations.

    Returns:
        State extracted of the input object.
    """
    assert by in ('reference', 'copy', 'deepcopy', 'clone', 'deepclone')
    by = by.replace('clone', 'copy')
    if device is not None:
        device = torch.device(device)

    # pylint: disable=import-outside-toplevel
    from torchopt.optim.meta.base import MetaOptimizer

    if by == 'reference':

        if device is not None:

            def replicate(t):
                return t.to(device=device)

        else:

            def replicate(t):
                return t

    elif by == 'copy':
        if device is not None:

            def replicate(t):
                return t.clone().to(device=device)

        else:

            def replicate(t):
                return t.clone()

    else:
        if device is not None:

            def replicate(t):
                return t.clone().detach_().to(device=device).requires_grad_(t.requires_grad)

        else:

            def replicate(t):
                return t.clone().detach_().requires_grad_(t.requires_grad)

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

        params = []

        def update_container(term):
            if len(term) != 0:
                params.append(
                    type(term)(
                        (k, replicate(v)) for k, v in term.items() if isinstance(v, torch.Tensor)
                    )
                )

        # pylint: disable=protected-access
        update_container(target._parameters)
        if with_buffer:
            update_container(target._buffers)
        for module in target.modules():
            if module is target:
                continue
            update_container(module._parameters)
            if with_buffer:
                update_container(module._buffers)
        return ModuleState(params=tuple(params), visual_contents=visual_contents)

    elif isinstance(target, MetaOptimizer):
        state = target.state_dict()

        def get_variable(t):
            if isinstance(t, torch.Tensor):
                return replicate(t)
            return t

        state = pytree.tree_map(get_variable, state)  # type: ignore[arg-type,assignment]
        return state

    raise RuntimeError(f'Unexpected class of {target}')


def _extract_container(
    module: nn.Module, with_buffer: bool = True
) -> Tuple[Dict[str, Optional[torch.Tensor]], ...]:
    if isinstance(module, nn.Module):
        containers = []

        def update_container(term):
            if len(term) != 0:
                containers.append(term)  # we need references to original dicts

        # pylint: disable=protected-access
        update_container(module._parameters)
        if with_buffer:
            update_container(module._buffers)
        for submodule in module.modules():
            if submodule is module:
                continue
            update_container(submodule._parameters)
            if with_buffer:
                update_container(submodule._buffers)
        return tuple(containers)

    raise RuntimeError(f'Unexpected class of {module}')


def recover_state_dict(
    target: Union[nn.Module, 'MetaOptimizer'],
    state: Union[ModuleState, Sequence['OptState']],
) -> None:
    """Recover state.

    This function is compatible for the ``extract_state``.

    Note that the recovering process is not in-place, so the tensors of the object will not be
    modified.

    Args:
        target: Target that need to recover.
        state: The recovering state.
    """
    # pylint: disable-next=import-outside-toplevel
    from torchopt.optim.meta.base import MetaOptimizer

    if isinstance(target, nn.Module):
        target_container = _extract_container(target)
        state = cast(ModuleState, state)
        for tgt, source in zip(target_container, state.params):
            tgt.update(source)
    elif isinstance(target, MetaOptimizer):
        state = cast(Sequence[OptState], state)
        target.load_state_dict(state)
    else:
        raise RuntimeError(f'Unexpected class of {target}')


def module_clone(
    target: Union[TensorTree, nn.Module, 'MetaOptimizer'],
    *,
    by: Literal['reference', 'copy', 'deepcopy'] = 'reference',  # type: ignore[name-defined]
    device: Optional[Union[int, str, torch.device]] = None,
) -> Union[TensorTree, nn.Module, 'MetaOptimizer']:
    """Clone a module.

    Args:
        target: The target to be cloned.
        by: The extract policy of tensors in the target.
            - :const:`'reference'`: The extracted tensors will be references to the original
            tensors.
            - :const:`'copy'`: The extracted tensors will be clones of the original tensors. This
            makes the copied tensors have :attr:`grad_fn` to be a ``<CloneBackward>`` function
            points to the original tensors.
            - :const:`'deepcopy'`: The extracted tensors will be deep-copied from the original
            tensors. The deep-copied tensors will detach from the original computation graph.
        device: If specified, move the cloned module to the specified device.

    Returns:
        The cloned module.
    """
    assert by in ('reference', 'copy', 'deepcopy', 'clone', 'deepclone')
    by = by.replace('clone', 'copy')
    if device is not None:
        device = torch.device(device)

    # pylint: disable=import-outside-toplevel
    import copy

    from torchopt.optim.meta.base import MetaOptimizer

    if isinstance(target, (nn.Module, MetaOptimizer)):
        cloned = copy.deepcopy(target)
        recover_state_dict(cloned, extract_state_dict(target, by=by, device=device))
        return cloned

    # Tree of tensors
    if by == 'reference':
        if device is not None:

            def replicate(t):
                return t.to(device=device)

        else:

            def replicate(t):
                return t

    elif by == 'copy':
        if device is not None:

            def replicate(t):
                return t.clone().to(device=device)

        else:

            def replicate(t):
                return t.clone()

    else:
        if device is not None:

            def replicate(t):
                return t.clone().detach_().to(device=device).requires_grad_(t.requires_grad)

        else:

            def replicate(t):
                return t.clone().detach_().requires_grad_(t.requires_grad)

    return pytree.tree_map(replicate, cast(TensorTree, target))


def module_detach_(
    target: Union['TensorTree', ModuleState, nn.Module, 'MetaOptimizer']
) -> Union['TensorTree', ModuleState, nn.Module, 'MetaOptimizer']:
    """Detach a module from the computation graph.

    Args:
        target: The target to be detached.

    Returns:
        The detached module.
    """
    stop_gradient(target)
    return target
