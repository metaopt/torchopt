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

import jax

import torch
from torch import nn

from typing import List, NamedTuple, Union, Dict
from TorchOpt._src.MetaOptimizer import MetaOptimizer


class _ModuleState(NamedTuple):
    params: List[Dict]

    visual_contents: Union[None, Dict] = None


def stop_gradient(target):
    """Stop the gradient for the input object.

  Since a tensor use `grad_fn` to connect itself with the previous computation
  graph, the back-propagated gradient will flow over the tensor and continue
  flow to the tensors that is connected by `grad_fn`. Some algorithms requires
  manually detaching tensors from the computation graph.

  Note that the stop_gradient operation is in-place.

  Args:
    target: the target that to be detached from the computation graph, it coule
      be a `nn.Module`, `TorchOpt.MetaOptimizer`, state of the
      `TorchOpt.MetaOptimizer`, or just a plain list of tensors.
    inplace: if True, the target will be detached in-place. if False, this function
      will return a detached copy of the target. The in-place operation is fast
      and memory efficient but may raise back-propagation error.
  """
    def f(obj):
        if isinstance(obj, torch.Tensor):
            requires_grad = obj.requires_grad
            obj.detach_().requires_grad_(requires_grad)
        return None

    if isinstance(target, _ModuleState):
        true_target = target.params
    elif isinstance(target, nn.Module):
        true_target = tuple(target.parameters())
    elif isinstance(target, MetaOptimizer):
        true_target, _ = jax.tree_flatten(target.state_dict())
    else:
        true_target = target

    jax.tree_map(f, true_target)


def extract_state_dict(mod, copy=False, *, with_buffer=True, enable_visual=False, visual_prefix=''):
    """Extract target state.

  Since a tensor use `grad_fn` to connect itself with the previous computation
  graph, the back-propagated gradient will flow over the tensor and continue
  flow to the tensors that is connected by `grad_fn`. Some algorithms requires
  manually detaching tensors from the computation graph.

  Note that the extracted state is a reference, which means any in-place operatior
  will affect the target that the state is extracted from.

  Args:
    mod: it coule be a `nn.Module` or `TorchOpt.MetaOptimizer`.
    with_buffer: extract buffer together with parameters, this argument is only
      used if the input target is `nn.Module`.
    enable_visual: add additional annoations, which could be used in computation
      graph visualization. Currently, this flag only has effect on `nn.Module` but
      we will support `TorchOpt.MetaOptimizer` later.
    visual_prefix: prefix for the visualization annoations.

  Returns:
    State extracted of the input object.
  """
    if isinstance(mod, nn.Module):
        if enable_visual:
            visual_contents = {}

            for k, v in mod.named_parameters():
                if v.grad_fn is not None:
                    visual_contents.update({v.grad_fn: (visual_prefix + k, v)})
                else:
                    visual_contents.update({v: visual_prefix + k})
        else:
            visual_contents = None

        params = []

        def get_v(v):
            if copy:
                requires_grad = v.requires_grad
                return v.clone().detach_().requires_grad_(requires_grad)
            else:
                return v

        def _update(term):
            if len(term) != 0:
                params.append(
                    {k: get_v(v) for k, v in term.items()})

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
            flatten_state, state_tree = jax.tree_flatten(state)

            def get_v(v):
                if not isinstance(v, torch.Tensor):
                    return v
                requires_grad = v.requires_grad
                return v.clone().detach_().requires_grad_(requires_grad)
            flatten_state = jax.tree_map(get_v, flatten_state)
            return state_tree.unflatten(flatten_state)
        else:
            return state

    else:
        raise RuntimeError(f"Unexpected class of {mod}")


def _extract_container(mod, with_buffer=True):
    if isinstance(mod, nn.Module):
        containers = []

        def _update(term):
            if len(term) != 0:
                containers.append(term)

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
    else:
        raise RuntimeError(f"Unexpected class of {mod}")


def recover_state_dict(mod, state):
    """Recover state.

  This function is compatiable for the `extract_state`.

  Note that the recovering process is not in-place, so the tensors of the object
  will not be modified.

  Args:
    mod: targe that need to recover.
    state: the recovering state.
  """
    if isinstance(mod, nn.Module):
        target_container = _extract_container(mod)
        for target, source in zip(target_container, state.params):
            target.update(source)
    elif isinstance(mod, MetaOptimizer):
        mod.load_state_dict(state)
    else:
        raise RuntimeError(f"Unexpected class of {mod}")
