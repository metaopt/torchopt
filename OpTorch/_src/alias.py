from functools import partial
from typing import Any, Optional, Union

from OpTorch._src import base
from OpTorch._src import combine
from OpTorch._src import transform
from .pytypes import ScalarOrSchedule


def _scale_by_lr(lr: ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(lr):
        return transform.scale_by_schedule(lambda count: m * lr(count))
    return transform.scale(m * lr)


def adam(
        lr: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        requires_grad: bool = False
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
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    the corresponding `GradientTransformation`.
  """
    return combine.chain(
        transform.scale_by_adam(
            b1=b1, b2=b2, eps=eps, eps_root=eps_root, requires_grad=requires_grad),
        _scale_by_lr(lr),
    )


meta_adam = partial(adam, requires_grad=True)


def custom_adam(
        lr: ScalarOrSchedule,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
        requires_grad: bool = False
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
    mu_dtype: optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    the corresponding `GradientTransformation`.
  """
    return combine.chain(
        transform.scale_by_custom_adam(
            lr=lr, b1=b1, b2=b2, eps=eps, eps_root=eps_root, requires_grad=requires_grad),
    )


def sgd(
        lr: ScalarOrSchedule,
        momentum: Optional[float] = None,
        nesterov: bool = False,
        requires_grad: bool = False,
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
    accumulator_dtype: optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation`.
  """
    return combine.chain(
        (transform.trace(decay=momentum, nesterov=nesterov,
                         requires_grad=requires_grad)
         if momentum is not None else base.identity()),
        _scale_by_lr(lr)
    )


meta_sgd = partial(sgd, requires_grad=True)
