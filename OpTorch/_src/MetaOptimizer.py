import torch
import jax

import OpTorch
from .alias import sgd, adam, custom_adam


class MetaOptimizer(object):
    def __init__(self, impl, net):
        self.impl = impl
        self.param_containers_groups = []
        self.state_groups = []
        self.add_param_group(net)

    def step(self, loss, *, hook=None):
        # step parameter only
        for idx, (state, param_containers) in enumerate(zip(self.state_groups, self.param_containers_groups)):
            flatten_params, containers_tree = jax.tree_util.tree_flatten(
                param_containers)
            grad = torch.autograd.grad(
                loss, flatten_params, create_graph=True, allow_unused=True)
            if hook is not None:
                def register_hook_fn(g):
                    return g.register_hook(hook) if g is not None else None
                jax.tree_map(register_hook_fn, grad)
            updates, state = self.impl.update(list(grad), state, False)
            self.state_groups[idx] = state
            new_params = OpTorch.apply_updates(
                flatten_params, list(updates), inplace=False)
            unflatten_new_params = containers_tree.unflatten(new_params)
            for (container, unflatten_param) in zip(param_containers, unflatten_new_params):
                container.update(unflatten_param)

    def add_param_group(self, net):
        from .utils import extract_state_dict
        net_state = extract_state_dict(net, with_buffer=False)
        flatten_param, _ = jax.tree_util.tree_flatten(net_state.containers)
        optim_state = self.impl.init(flatten_param)
        self.state_groups.append(optim_state)
        self.param_containers_groups.append(net_state.containers)

    def state_dict(self):
        out_groups = tuple(group for group in self.state_groups)
        return out_groups

    def load_state_dict(self, state_dict):
        self.state_groups = list(group for group in state_dict)


class MetaAdam(MetaOptimizer):
    def __init__(self, net, *args, **kwargs):
        super().__init__(adam(*args, **kwargs, requires_grad=True), net)


class CustomMetaAdam(MetaOptimizer):
    def __init__(self, net, *args, **kwargs):
        super().__init__(custom_adam(*args, **kwargs, requires_grad=True), net)


class MetaSGD(MetaOptimizer):
    def __init__(self, net, *args, **kwargs):
        super().__init__(sgd(*args, **kwargs, requires_grad=True), net)
