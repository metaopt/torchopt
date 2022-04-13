import jax

import torch
from torch import nn
from torch import Tensor

from typing import List, NamedTuple, Union, Optional, Dict
from OpTorch import MetaOptimizer


class _ModuleState(NamedTuple):
    params: List[Dict]

    containers: List[Union[Dict[str, Optional[nn.Parameter]],
                           Dict[str, Optional[Tensor]]]] = None

    visual_contents: Union[None, Dict] = None


def stop_gradient(target, inplace=True):
    """Stop the gradient for the input object.

  Since a tensor use `grad_fn` to connect itself with the previous computation
  graph, the back-propagated gradient will flow over the tensor and continue
  flow to the tensors that is connected by `grad_fn`. Some algorithms requires
  manually detaching tensors from the computation graph.

  Note that the stop_gradient operation is in-place.

  Args:
    target: the target that to be detached from the computation graph, it coule
    be a `nn.Module`, `OpTorch.MetaOptimizer`, state of the
    `OpTorch.MetaOptimizer`, or just a plain list of tensors.
  """
    if inplace:
        def f(obj):
            if isinstance(obj, torch.Tensor):
                requires_grad = obj.requires_grad
                obj.detach_().requires_grad_(requires_grad)
            return None
    else:
        def f(obj):
            if isinstance(obj, torch.Tensor):
                requires_grad = obj.requires_grad
                obj = obj.clone().detach_().requires_grad_(requires_grad)
            return obj

    if isinstance(target, _ModuleState):
        true_target = target.params
    else:
        true_target = target

    if inplace:
        jax.tree_map(f, true_target)
        return None
    else:
        flatten_target, target_tree = jax.tree_flatten(true_target)
        detached_flatten_target = jax.tree_map(f, flatten_target)
        detached_target = target_tree.unflatten(detached_flatten_target)
        if isinstance(target, _ModuleState):
            return _ModuleState(params=detached_target)
        else:
            return detached_target


def extract_state_dict(mod, *, with_buffer=True, enable_visual=False, visual_prefix=''):
    """Extract target state.

  Since a tensor use `grad_fn` to connect itself with the previous computation
  graph, the back-propagated gradient will flow over the tensor and continue
  flow to the tensors that is connected by `grad_fn`. Some algorithms requires
  manually detaching tensors from the computation graph.

  Note that the extracted state is a reference, which means any in-place operatior
  will affect the target that the state is extracted from.

  Args:
    mod: it coule be a `nn.Module` or `OpTorch.MetaOptimizer`.
    with_buffer: extract buffer together with parameters, this argument is only
    used if the input target is `nn.Module`.
    enable_visual: add additional annoations, which could be used in computation
    graph visualization. Currently, this flag only has effect on `nn.Module` but
    we will support `OpTorch.MetaOptimizer` later.
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

        containers = []
        params = []

        def _update(term):
            if len(term) != 0:
                containers.append(term)
                params.append({k: v for k, v in term.items()})

        _update(mod._parameters)
        if with_buffer:
            _update(mod._buffers)
        for module in mod.modules():
            _update(module._parameters)
            if with_buffer:
                _update(module._buffers)
        return _ModuleState(containers=containers, params=params, visual_contents=visual_contents)
    elif isinstance(mod, MetaOptimizer):
        return mod.state_dict()
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
        target_state = extract_state_dict(mod)
        for target, source in zip(target_state.containers, state.params):
            target.update(source)
    elif isinstance(mod, MetaOptimizer):
        mod.load_state_dict(state)
    else:
        raise RuntimeError(f"Unexpected class of {mod}")
