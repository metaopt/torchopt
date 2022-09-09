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

from typing import Any, Callable, Optional, Tuple, Union

from torchopt._src import base, combine, transform
from torchopt._src.typing import ScalarOrSchedule


def _flip_sign_and_weight_decay(weight_decay: float = 0.0, maximize=False):
    if not 0.0 <= weight_decay:  # pylint: disable=unneeded-not
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')

    if not maximize and weight_decay == 0.0:
        return base.identity()

    def init_fn(params):  # pylint: disable=unused-argument
        return base.EmptyState()

    if not maximize:  # gradient descent

        def update_fn(updates, state, *, params=None, inplace=True):
            assert params is not None, (
                'Parameters are required for weight decay. '
                'Call `update(updates, state, params=params)` instead.'
            )

            if inplace:

                def f(g, p):
                    if g is not None:
                        if g.requires_grad:
                            return g.add_(p, alpha=weight_decay)
                        return g.add_(p.data, alpha=weight_decay)
                    return None

            else:

                def f(g, p):
                    return g.add(p, alpha=weight_decay) if g is not None else None

            updates = transform.map_flattened(f, updates, params)
            return updates, state

    else:  # gradient ascent

        if weight_decay == 0.0:
            # pylint: disable-next=unused-argument
            def update_fn(updates, state, *, params=None, inplace=True):
                if inplace:

                    def f(g):
                        return g.neg_() if g is not None else None

                else:

                    def f(g):
                        return g.neg() if g is not None else None

                updates = transform.map_flattened(f, updates)
                return updates, state

        else:

            def update_fn(updates, state, *, params=None, inplace=True):
                assert params is not None, (
                    'Parameters are required for weight decay. '
                    'Call `update(updates, state, params=params)` instead.'
                )

                if inplace:

                    def f(g, p):
                        if g is not None:
                            if g.requires_grad:
                                return g.neg_().add_(p, alpha=weight_decay)
                            return g.neg_().add_(p.data, alpha=weight_decay)
                        return None

                else:

                    def f(g, p):
                        return g.neg().add_(p, alpha=weight_decay) if g is not None else None

                updates = transform.map_flattened(f, updates, params)
                return updates, state

    return base.GradientTransformation(init_fn, update_fn)


def _scale_by_neg_lr(lr: ScalarOrSchedule):
    if not (callable(lr) or 0.0 <= lr):
        raise ValueError(f'Invalid learning rate: {lr}')

    if callable(lr):

        def schedule_wrapper(count):
            return -lr(count)  # type: ignore[operator]

        # pylint: disable-next=protected-access
        return transform._scale_by_schedule(schedule_wrapper, already_flattened=True)
    return transform._scale(-lr, already_flattened=True)  # pylint: disable=protected-access


# pylint: disable-next=too-many-arguments
def adam(
    lr: ScalarOrSchedule = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    *,
    eps_root: float = 0.0,
    moment_requires_grad: bool = False,
    maximize: bool = False,
    use_accelerated_op: bool = False,
) -> base.GradientTransformation:
    """The functional Adam optimizer.

    Adam is an SGD variant with learning rate adaptation. The *learning rate* used for each weight
    is computed from estimates of first- and second-order moments of the gradients (using suitable
    exponential moving averages).

    References:
        - Kingma et al, 2014: https://arxiv.org/abs/1412.6980

    Args:
        lr: (default: :const:`1e-3`)
            This is a fixed global scaling factor.
        betas: (default: :const:`(0.9, 0.999)`)
            Coefficients used for computing running averages of gradient and its square.
        eps: (default: :const:`1e-8`)
            A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        weight_decay: (default: :const:`0.0`)
            Weight decay, add L2 penalty to parameters.
        eps_root: (default: :data:`0.0`)
            A small constant applied to denominator inside the square root (as in RMSProp), to avoid
            dividing by zero when rescaling. This is needed for example when computing
            (meta-)gradients through Adam.
        moment_requires_grad: (default: :data:`False`)
            If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
            flag is often used in Meta-Learning algorithms.
        maximize: (default: :data:`False`)
            Maximize the params based on the objective, instead of minimizing.
        use_accelerated_op: (default: :data:`False`)
            If :data:`True` use our implemented fused operator.

    Returns:
        The corresponding :class:`GradientTransformation` instance.
    """
    b1, b2 = betas
    # pylint: disable=unneeded-not
    if not (callable(lr) or 0.0 <= lr):
        raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= b1 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 0: {b1}')
    if not 0.0 <= b2 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 1: {b2}')
    if not 0.0 <= weight_decay:
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    # pylint: enable=unneeded-not

    if use_accelerated_op:
        adam_scaler = transform._scale_by_accelerated_adam  # pylint: disable=protected-access
    else:
        adam_scaler = transform._scale_by_adam  # pylint: disable=protected-access

    return transform.with_flattened_tree(
        combine.chain(
            _flip_sign_and_weight_decay(weight_decay=weight_decay, maximize=maximize),
            adam_scaler(
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                moment_requires_grad=moment_requires_grad,
                already_flattened=True,
            ),
            _scale_by_neg_lr(lr),
        )
    )


# pylint: disable-next=too-many-arguments
def adamw(
    lr: ScalarOrSchedule = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    *,
    eps_root: float = 0.0,
    mask: Optional[Union[Any, Callable[['base.Params'], Any]]] = None,
    moment_requires_grad: bool = False,
    maximize: bool = False,
    use_accelerated_op: bool = False,
) -> base.GradientTransformation:
    """Adam with weight decay regularization.

    AdamW uses weight decay to regularize learning towards small weights, as
    this leads to better generalization. In SGD you can also use L2 regularization
    to implement this as an additive loss term, however L2 regularization
    does not behave as intended for adaptive gradient algorithms such as Adam.

    References:
        - Loshchilov et al, 2019: https://arxiv.org/abs/1711.05101

    Args:
        lr: (default: :const:`1e-3`)
            This is a fixed global scaling factor.
        betas: (default: :const:`(0.9, 0.999)`)
            Coefficients used for computing running averages of gradient and its square.
        eps: (default: :const:`1e-8`)
            A small constant applied to denominator outside of the square root (as in the Adam
            paper) to avoid dividing by zero when rescaling.
        weight_decay: (default: :const:`1e-2`)
            Strength of the weight decay regularization. Note that this weight decay is multiplied
            with the learning rate. This is consistent with other frameworks such as PyTorch, but
            different from (Loshchilov et al, 2019) where the weight decay is only multiplied with
            the "schedule multiplier", but not the base learning rate.
        eps_root: (default: :data:`0.0`)
            A small constant applied to denominator inside the square root (as in RMSProp), to avoid
            dividing by zero when rescaling. This is needed for example when computing
            (meta-)gradients through Adam.
        mask: (default: :data:`None`)
            A tree with same structure as (or a prefix of) the params PyTree, or a Callable that
            returns such a pytree given the params/updates. The leaves should be booleans,
            :data:`True` for leaves/subtrees you want to apply the weight decay to, and
            :data:`False` for those you want to skip. Note that the Adam gradient
            transformations are applied to all parameters.
        moment_requires_grad: (default: :data:`False`)
            If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
            flag is often used in Meta-Learning algorithms.
        maximize: (default: :data:`False`)
            Maximize the params based on the objective, instead of minimizing.
        use_accelerated_op: (default: :data:`False`)
            If :data:`True` use our implemented fused operator.

    Returns:
        The corresponding :class:`GradientTransformation` instance.
    """
    b1, b2 = betas
    # pylint: disable=unneeded-not
    if not (callable(lr) or 0.0 <= lr):
        raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= b1 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 0: {b1}')
    if not 0.0 <= b2 < 1.0:
        raise ValueError(f'Invalid beta parameter at index 1: {b2}')
    if not 0.0 <= weight_decay:
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    # pylint: enable=unneeded-not

    if use_accelerated_op:
        adam_scaler = transform._scale_by_accelerated_adam  # pylint: disable=protected-access
    else:
        adam_scaler = transform._scale_by_adam  # pylint: disable=protected-access

    return transform.with_flattened_tree(
        combine.chain(
            _flip_sign_and_weight_decay(weight_decay=0.0, maximize=maximize),
            adam_scaler(
                b1=b1,
                b2=b2,
                eps=eps,
                eps_root=eps_root,
                moment_requires_grad=moment_requires_grad,
                already_flattened=True,
            ),
            transform._add_decayed_weights(  # pylint: disable=protected-access
                weight_decay=weight_decay,
                mask=mask,
                already_flattened=True,
            ),
            _scale_by_neg_lr(lr),
        )
    )


# pylint: disable-next=too-many-arguments
def rmsprop(
    lr: ScalarOrSchedule = 1e-2,
    alpha: float = 0.99,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    momentum: float = 0.0,
    centered: bool = False,
    *,
    initial_scale: float = 0.0,
    nesterov: bool = False,
    maximize: bool = False,
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
        lr: (default: :const:`1e-2`)
            This is a fixed global scaling factor.
        alpha: (default: :const:`0.99`)
            Smoothing constant, the decay used to track the magnitude of previous gradients.
        eps: (default: :const:`1e-8`)
            A small numerical constant to avoid dividing by zero when rescaling.
        weight_decay: (default: :const:`0.0`)
            Weight decay, add L2 penalty to parameters.
        momentum: (default: :const:`0.0`)
            The decay rate used by the momentum term. The momentum is not used when it is set to
            :const:`0.0`.
        centered: (default: :data:`False`)
            If :data:`True`, use the variance of the past gradients to rescale the latest
            gradients.
        initial_scale: (default: :data:`0.0`)
            Initialization of accumulators tracking the magnitude of previous updates. PyTorch
            uses :data:`0.0`, TensorFlow 1.x uses :data:`1.0`. When reproducing results from a
            paper, verify the value used by the authors.
        nesterov: (default: :data:`False`)
            Whether to use Nesterov momentum.
        maximize: (default: :data:`False`)
            Maximize the params based on the objective, instead of minimizing.

    Returns:
        The corresponding :class:`GradientTransformation` instance.
    """
    # pylint: disable=unneeded-not
    if not (callable(lr) or 0.0 <= lr):
        raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= alpha:
        raise ValueError(f'Invalid alpha value: {alpha}')
    if not 0.0 <= eps:
        raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= momentum:
        raise ValueError(f'Invalid momentum value: {momentum}')
    if not 0.0 <= weight_decay:
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    # pylint: enable=unneeded-not

    if centered:
        rmsprop_scaler = transform._scale_by_stddev  # pylint: disable=protected-access
    else:
        rmsprop_scaler = transform._scale_by_rms  # pylint: disable=protected-access

    return transform.with_flattened_tree(
        combine.chain(
            _flip_sign_and_weight_decay(weight_decay=weight_decay, maximize=maximize),
            rmsprop_scaler(
                alpha=alpha,
                eps=eps,
                initial_scale=initial_scale,
                already_flattened=True,
            ),
            transform._trace(  # pylint: disable=protected-access
                momentum=momentum,
                nesterov=nesterov,
                already_flattened=True,
            ),
            _scale_by_neg_lr(lr),
        )
    )


def sgd(
    lr: ScalarOrSchedule,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    *,
    moment_requires_grad: bool = False,
    maximize: bool = False,
) -> base.GradientTransformation:
    """The functional version of the canonical Stochastic Gradient Descent optimizer.

    This implements stochastic gradient descent. It also includes support for momentum, and nesterov
    acceleration, as these are standard practice when using stochastic gradient descent to train
    deep neural networks.

    References:
        - Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

    Args:
        lr: This is a fixed global scaling factor.
        momentum: (default: :const:`0.0`)
            The decay rate used by the momentum term. The momentum is not used when it is set to
            :const:`0.0`.
        weight_decay: (default: :const:`0.0`)
            Weight decay, add L2 penalty to parameters.
        dampening: (default: :const:`0.0`)
            Dampening for momentum.
        nesterov: (default: :data:`False`)
            Whether to use Nesterov momentum.
        moment_requires_grad: (default: :data:`False`)
            If :data:`True` the momentums will be created with flag ``requires_grad=True``, this
            flag is often used in Meta-Learning algorithms.
        maximize: (default: :data:`False`)
            Maximize the params based on the objective, instead of minimizing.

    Returns:
        The corresponding :class:`GradientTransformation` instance.
    """
    # pylint: disable=unneeded-not
    if not (callable(lr) or 0.0 <= lr):
        raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= momentum:
        raise ValueError(f'Invalid momentum value: {momentum}')
    if not 0.0 <= weight_decay:
        raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        raise ValueError('Nesterov momentum requires a momentum and zero dampening')
    # pylint: enable=unneeded-not

    return transform.with_flattened_tree(
        combine.chain(
            _flip_sign_and_weight_decay(weight_decay=weight_decay, maximize=maximize),
            transform._trace(  # pylint: disable=protected-access
                momentum=momentum,
                dampening=dampening,
                nesterov=nesterov,
                moment_requires_grad=moment_requires_grad,
                already_flattened=True,
            ),
            _scale_by_neg_lr(lr),
        )
    )
