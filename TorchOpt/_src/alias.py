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
# This file is modified from:
# https://github.com/deepmind/optax/blob/master/optax/_src/alias.py
# ==============================================================================
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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

from typing import Optional
import jax

from TorchOpt._src import base
from TorchOpt._src import combine
from TorchOpt._src import transform
from TorchOpt._src.pytypes import ScalarOrSchedule


def _scale_by_lr(lr: ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(lr):
        def schedule_wrapper(count):
            def f(scaled_lr): return m * scaled_lr
            return jax.tree_map(f, lr(count))
        return transform.scale_by_schedule(schedule_wrapper)
    return transform.scale(m * lr)


def adam(
        lr: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        moment_requires_grad: bool = False,
        use_accelerated_op: bool = False
) -> base.GradientTransformation:
    """The classic Adam optimiser.

  Adam is an SGD variant with learning rate adaptation. The `lr`
  used for each weight is computed from estimates of first- and second-order
  moments of the gradients (using suitable exponential moving averages).

  References:
    Kingma et al, 2014: https://arxiv.org/abs/1412.6980

  Args:
    lr: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    b2: the exponential decay rate to track the second moment of past gradients.
    eps: a small constant applied to denominator outside of the square root
      (as in the Adam paper) to avoid dividing by zero when rescaling.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for example when computing (meta-)gradients through Adam.
    moment_requires_grad: (default `False`), if True the momentums will be created with flag
      `requires_grad=True`, this flag is often used in Meta Learning algorithms.
    use_accelerated_op: (default `False`), if True use our implemented fused operator.

  Returns:
    the corresponding `GradientTransformation`.
  """
    adam_inst = transform.scale_by_accelerated_adam if use_accelerated_op else transform.scale_by_adam
    return combine.chain(
        adam_inst(b1=b1, b2=b2, eps=eps, eps_root=eps_root,
                  moment_requires_grad=moment_requires_grad),
        _scale_by_lr(lr),
    )


def sgd(
        lr: ScalarOrSchedule,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        moment_requires_grad: bool = False,
) -> base.GradientTransformation:
    """A canonical Stochastic Gradient Descent optimiser.

  This implements stochastic gradient descent. It also includes support for
  momentum, and nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

  Args:
    lr: this is a fixed global scaling factor.
    momentum: (default `None`), the `decay` rate used by the momentum term,
      when it is set to `None`, then momentum is not used at all.
    nesterov (default `False`): whether nesterov momentum is used.
    moment_requires_grad: (default `False`), if True the momentums will be created with flag
      `requires_grad=True`, this flag is often used in Meta Learning algorithms.

  Returns:
    A `GradientTransformation`.
  """
    return combine.chain(
        (transform.trace(decay=momentum, nesterov=nesterov,
                         moment_requires_grad=moment_requires_grad)
         if momentum is not None else base.identity()),
        _scale_by_lr(lr)
    )
