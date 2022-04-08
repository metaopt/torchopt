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

import torch
import jax
from TorchOpt._src.pytypes import ScalarOrSchedule
from TorchOpt._src.update import apply_updates
from TorchOpt._src.alias import adam, sgd


class Optimizer(object):
    """A high-level base class that has the similar with `torch.optim.Optimier`"""

    def __init__(self, params, impl):
        """
        Args:
          params (iterable): an iterable of `torch.Tensor`s. Specifies what Tensors should be optimized.
          impl (base.GradientTransformation): a low level optimizer function, it could be 
            a optimizer function provided by `alias.py` or a customerized `chain` provided by 
            `combine.py`. Note that use `MetaOptimizer(sgd())` or `MetaOptimizer(chain(sgd())) 
            is equavalent to `SGD`.

        """
        if not isinstance(params, list):
            params = list(params)
        self.impl = impl
        self.param_groups = []
        self.param_tree_groups = []
        self.state_groups = []
        self.add_param_group(params)

    def zero_grad(self, set_to_none: bool = False):
        """Sets the gradients of all optimized `torch.Tensor`s to zero.

        The behivour is similar to `torch.optim.Optimizer.zero_grad`.

        Args:
          set_to_none (bool): instead of setting to zero, set the grads to None.

        """
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

    def step(self, closure=None):
        """Performs a single optimization step (parameter update).

        The behivour is similar to `torch.optim.Optimizer.step`.

        Args:
          closure (callable, optional): A closure that reevaluates the model and returns the loss.

        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param, state in zip(self.param_groups, self.state_groups):
            def f(p): return p.grad
            grad = jax.tree_map(f, param)
            updates, _ = self.impl.update(grad, state)
            apply_updates(param, updates)

        return loss

    def add_param_group(self, params):
        params, tree = jax.tree_flatten(params)
        params = tuple(params)
        self.param_groups.append(params)
        self.param_tree_groups.append(tree)
        self.state_groups.append(self.impl.init(params))


class SGD(Optimizer):
    """The classic Adam optimiser."""

    def __init__(self,
                 params,
                 lr: ScalarOrSchedule,
                 momentum: float = None,
                 nesterov: bool = False):
        """
        Args:
          params (iterable): an iterable of `torch.Tensor`s. Specifies what Tensors should be optimized.
          args: other arguments see `alias.adam`.
        """
        super().__init__(params, sgd(lr=lr,
                                     momentum=momentum,
                                     nesterov=nesterov,
                                     moment_requires_grad=False))


class Adam(Optimizer):
    """A canonical Stochastic Gradient Descent optimiser."""

    def __init__(self,
                 params,
                 lr: ScalarOrSchedule,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 eps: float = 1e-8,
                 eps_root: float = 0.0,
                 use_accelerated_op: bool = False):
        """
        Args:
          params (iterable): an iterable of `torch.Tensor`s. Specifies what Tensors should be optimized.
          args: other arguments see `alias.sgd`.
        """
        super().__init__(params, adam(lr=lr,
                                      b1=b1,
                                      b2=b2,
                                      eps=eps,
                                      eps_root=eps_root,
                                      moment_requires_grad=False,
                                      use_accelerated_op=use_accelerated_op))
