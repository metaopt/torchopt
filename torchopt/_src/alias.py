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

# pylint: disable=invalid-name

from typing import Any, Callable, Optional, Union

import jax

from torchopt._src import base, combine, transform
from torchopt._src.typing import ScalarOrSchedule


def _scale_by_lr(lr: ScalarOrSchedule, flip_sign=True):
    sign = -1 if flip_sign else 1
    if callable(lr):

        def schedule_wrapper(count):
            def f(scaled_lr):
                return sign * scaled_lr

            return jax.tree_map(f, lr(count))  # type: ignore

        return transform.scale_by_schedule(schedule_wrapper)
    return transform.scale(sign * lr)


# pylint: disable=too-many-arguments
def adam(
    lr: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
    use_accelerated_op: bool = False,
) -> base.GradientTransformation:
    """The functional Adam optimizer.

    Adam is an SGD variant with learning rate adaptation. The *learning rate* used for each weight
    is computed from estimates of first- and second-order moments of the gradients (using suitable
    exponential moving averages).

    References:
        - Kingma et al, 2014: https://arxiv.org/abs/1412.6980

    Args:
        lr: This is a fixed global scaling factor.
        b1: The exponential decay rate to track the first moment of past gradients.
        b2: The exponential decay rate to track the second moment of past gradients.
        eps:
            A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        eps_root: (default: :data:`0.0`)
            A small constant applied to denominator inside the square root (as in RMSProp), to avoid
            dividing by zero when rescaling. This is needed for example when computing
            (meta-)gradients through Adam.
        moment_requires_grad: (default: :data:`False`)
            If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
            flag is often used in Meta Learning algorithms.
        use_accelerated_op: (default: :data:`False`)
            If :data:`True` use our implemented fused operator.

    Returns:
        The corresponding :class:`GradientTransformation` instance.
    """
    adam_inst = (
        transform.scale_by_accelerated_adam if use_accelerated_op else transform.scale_by_adam
    )
    return combine.chain(
        adam_inst(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            moment_requires_grad=moment_requires_grad,
        ),
        _scale_by_lr(lr),
    )


def sgd(
    lr: ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    moment_requires_grad: bool = False,
) -> base.GradientTransformation:
    """The functional version of the canonical Stochastic Gradient Descent optimizer.

    This implements stochastic gradient descent. It also includes support for momentum, and nesterov
    acceleration, as these are standard practice when using stochastic gradient descent to train
    deep neural networks.

    References:
        - Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

    Args:
        lr: This is a fixed global scaling factor.
        momentum: (default: :data:`None`)
            The ``decay`` rate used by the momentum term, when it is set to :data:`None`, then
            momentum is not used at all.
        nesterov: (default: :data:`False`)
            Whether the nesterov momentum is used.
        moment_requires_grad: (default: :data:`False`)
            If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
            flag is often used in Meta-Learning algorithms.

    Returns:
        A :class:`GradientTransformation` instance.
    """
    return combine.chain(
        (
            transform.trace(
                decay=momentum,
                nesterov=nesterov,
                moment_requires_grad=moment_requires_grad,
            )
            if momentum is not None
            else base.identity()
        ),
        _scale_by_lr(lr),
    )


# pylint: disable=too-many-arguments
def rmsprop(
    lr: ScalarOrSchedule,
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.0,
    centered: bool = False,
    momentum: Optional[float] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
    """The functional version of the RMSProp optimizer.

    RMSProp is an SGD variant with learning rate adaptation. The *learning rate* used for each
    weight is scaled by a suitable estimate of the magnitude of the gradients on previous steps.
    Several variants of RMSProp can be found in the literature. This alias provides an easy to
    configure RMSProp optimizer that can be used to switch between several of these variants.

    References:
        - Tieleman and Hinton, 2012: http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
        - Graves, 2013: https://arxiv.org/abs/1308.0850

    Args:
        lr: This is a fixed global scaling factor.
        decay: The decay used to track the magnitude of previous gradients.
        eps: A small numerical constant to avoid dividing by zero when rescaling.
        initial_scale: (default: :data:`0.0`)
            Initialization of accumulators tracking the magnitude of previous updates. PyTorch uses
            :data:`0.0`, TensorFlow 1.x uses :data:`1.0`. When reproducing results from a paper,
            verify the value used by the authors.
        centered: (default: :data:`False`)
            Whether the second moment or the variance of the past gradients is used to rescale the
            latest gradients.
        momentum: (default: :data:`None`)
            The ``decay`` rate used by the momentum term, when it is set to :data:`None`, then
            momentum is not used at all.
        nesterov: (default: :data:`False`)
            Whether the nesterov momentum is used.

    Returns:
        The corresponding :class:`GradientTransformation` instance.
    """
    if centered:
        return combine.chain(
            transform.scale_by_stddev(decay=decay, eps=eps, initial_scale=initial_scale),
            _scale_by_lr(lr),
            (
                transform.trace(decay=momentum, nesterov=nesterov)
                if momentum is not None
                else base.identity()
            ),
        )

    return combine.chain(
        transform.scale_by_rms(decay=decay, eps=eps, initial_scale=initial_scale),
        _scale_by_lr(lr),
        (
            transform.trace(decay=momentum, nesterov=nesterov)
            if momentum is not None
            else base.identity()
        ),
    )


# pylint: disable=too-many-arguments
def adamw(
    lr: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
    use_accelerated_op: bool = False,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
    """Adam with weight decay regularization.

    AdamW uses weight decay to regularise learning towards small weights, as
    this leads to better generalisation. In SGD you can also use L2 regularisation
    to implement this as an additive loss term, however L2 regularization
    does not behave as intended for adaptive gradient algorithms such as Adam.

    References:
        - Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

    Args:
        learning_rate: this is a fixed global scaling factor.
        b1: the exponential decay rate to track the first moment of past gradients.
        b2: the exponential decay rate to track the second moment of past gradients.
        eps: a small constant applied to denominator outside of the square root
            (as in the Adam paper) to avoid dividing by zero when rescaling.
            eps_root: (default `0`), a small constant applied to denominator inside the
            square root (as in RMSProp), to avoid dividing by zero when rescaling.
            This is needed for instance when computing (meta-)gradients through Adam.
        mu_dtype: optional `dtype` to be used for the first order accumulator; if
            `None` then the `dtype` is inferred from `params` and `updates`.
        weight_decay: strength of the weight decay regularization. Note that this
            weight decay is multiplied with the learning rate. This is consistent
            with other frameworks such as PyTorch, but different from
            (Loshchilov et al, 2019) where the weight decay is only multiplied with
            the "schedule multiplier", but not the base learning rate.
        mask: a tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the weight decay to, and `False` for those you want to skip. Note
            that the Adam gradient transformations are applied to all parameters.

    Returns:
      the corresponding `GradientTransformation`.
    """
    adam_inst = (
        transform.scale_by_accelerated_adam if use_accelerated_op else transform.scale_by_adam
    )
    return combine.chain(
        adam_inst(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            moment_requires_grad=moment_requires_grad,
        ),
        transform.add_decayed_weights(weight_decay, mask),
        _scale_by_lr(lr),
    )
