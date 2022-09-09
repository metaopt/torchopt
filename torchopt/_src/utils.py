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

from typing import Dict, List, NamedTuple, Union

import optree as pytree
import torch
import torch.nn as nn


class _ModuleState(NamedTuple):
    params: List[Dict]
    visual_contents: Union[None, Dict] = None


# mypy: ignore-errors
def stop_gradient(target):
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
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from torchopt._src.optimizer.meta.base import MetaOptimizer

    def f(obj):
        if torch.is_tensor(obj):
            requires_grad = obj.requires_grad
            obj.detach_().requires_grad_(requires_grad)

    if isinstance(target, _ModuleState):
        true_target = target.params
    elif isinstance(target, nn.Module):
        true_target = tuple(target.parameters())
    elif isinstance(target, MetaOptimizer):
        true_target = pytree.tree_leaves(target.state_dict())
    else:
        true_target = target

    pytree.tree_map(f, true_target)


# pylint: disable-next=too-many-branches,too-many-locals
def extract_state_dict(mod, copy=False, *, with_buffer=True, enable_visual=False, visual_prefix=''):
    """Extract target state.

    Since a tensor use :attr:`grad_fn` to connect itself with the previous computation graph, the
    back-propagated gradient will flow over the tensor and continue flow to the tensors that is
    connected by :attr:`grad_fn`. Some algorithms requires manually detaching tensors from the
    computation graph.

    Note that the extracted state is a reference, which means any in-place operator will affect the
    target that the state is extracted from.

    Args:
        mod: It could be a :class:`nn.Module` or :class:`torchopt.MetaOptimizer`.
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
    # pylint: disable=import-outside-toplevel,cyclic-import
    from torchopt._src.optimizer.meta.base import MetaOptimizer

    if isinstance(mod, nn.Module):  # pylint: disable=no-else-return
        if enable_visual:
            visual_contents = {}

            for k, v in mod.named_parameters():  # pylint: disable=invalid-name
                if v.grad_fn is not None:
                    visual_contents.update({v.grad_fn: (visual_prefix + k, v)})
                else:
                    visual_contents.update({v: visual_prefix + k})
        else:
            visual_contents = None

        params = []

        def get_variable(t):
            if copy:
                requires_grad = t.requires_grad
                return t.clone().detach_().requires_grad_(requires_grad)
            return t

        def _update(term):
            if len(term) != 0:
                params.append({k: get_variable(v) for k, v in term.items()})

        # pylint: disable=protected-access
        _update(mod._parameters)
        if with_buffer:
            _update(mod._buffers)
        for module in mod.modules():
            if module is mod:
                continue
            _update(module._parameters)
            if with_buffer:
                _update(module._buffers)
        return _ModuleState(params=tuple(params), visual_contents=visual_contents)

    elif isinstance(mod, MetaOptimizer):
        state = mod.state_dict()
        if copy:

            def get_variable(t):
                if not torch.is_tensor(t):
                    return t
                requires_grad = t.requires_grad
                return t.clone().detach_().requires_grad_(requires_grad)

            return pytree.tree_map(get_variable, state)

        return state

    raise RuntimeError(f'Unexpected class of {mod}')


def _extract_container(mod, with_buffer=True):
    if isinstance(mod, nn.Module):
        containers = []

        def _update(term):
            if len(term) != 0:
                containers.append(term)

        # pylint: disable=protected-access
        _update(mod._parameters)
        if with_buffer:
            _update(mod._buffers)
        for module in mod.modules():
            if module is mod:
                continue
            _update(module._parameters)
            if with_buffer:
                _update(module._buffers)
        return tuple(containers)

    raise RuntimeError(f'Unexpected class of {mod}')


def recover_state_dict(mod, state):
    """Recover state.

    This function is compatible for the ``extract_state``.

    Note that the recovering process is not in-place, so the tensors of the object will not be
    modified.

    Args:
        mod: Target that need to recover.
        state: The recovering state.
    """
    # pylint: disable-next=import-outside-toplevel,cyclic-import
    from torchopt._src.optimizer.meta.base import MetaOptimizer

    if isinstance(mod, nn.Module):
        target_container = _extract_container(mod)
        for target, source in zip(target_container, state.params):
            target.update(source)
    elif isinstance(mod, MetaOptimizer):
        mod.load_state_dict(state)
    else:
        raise RuntimeError(f'Unexpected class of {mod}')
