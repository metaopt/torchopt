import jax
from .update import apply_updates
from .alias import custom_adam, adam, sgd


class Optimizer(object):
    def __init__(self, impl, params):
        if not isinstance(params, list):
            params = list(params)
        self.impl = impl
        self.param_groups = []
        self.param_tree_groups = []
        self.state_groups = []
        self.add_param_group(params)

    def zero_grad(self, set_to_none: bool = False):
        for group in self.param_groups:
            if set_to_none:
                def f(p):
                    p.grad = None
                    return None
            else:
                def f(p):
                    if p.grad is None:
                        return None
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()
                    return None
            jax.tree_map(f, group)

    def state_dict(self):
        return self.state_groups

    def load_state_dict(self, state_dict):
        self.state_groups = state_dict

    def step(self):
        for param, state in zip(self.param_groups, self.state_groups):
            def f(p): return p.grad
            grad = jax.tree_map(f, param)
            updates, _ = self.impl.update(grad, state)
            apply_updates(param, list(updates))

    def add_param_group(self, params):
        params, tree = jax.tree_flatten(params)
        self.param_groups.append(params)
        self.param_tree_groups.append(tree)
        self.state_groups.append(self.impl.init(params))


class SGD(Optimizer):
    def __init__(self, params, *args, **kwargs):
        super().__init__(sgd(*args, **kwargs), params)


class Adam(Optimizer):
    def __init__(self, params, *args, **kwargs):
        super().__init__(adam(*args, **kwargs), params)


class CustomAdam(Optimizer):
    def __init__(self, params, *args, **kwargs):
        super().__init__(custom_adam(*args, **kwargs), params)
