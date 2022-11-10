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
"""Zero-Order Gradient Estimation."""

import functools
import sys as _sys
from types import ModuleType as _ModuleType
from typing import Any, Callable, List, Sequence, Tuple, Union
from typing_extensions import Literal  # Python 3.8+
from typing_extensions import Protocol  # Python 3.8+
from typing_extensions import TypeAlias  # Python 3.10+

import torch
from torch.autograd import Function
from torch.distributions import Distribution

from torchopt import pytree
from torchopt.typing import ListOfTensors, Numeric, TupleOfOptionalTensors


class Samplable(Protocol):  # pylint: disable=too-few-public-methods
    """Abstract protocol class that supports sampling."""

    def sample(
        self, sample_shape: torch.Size = torch.Size()  # pylint: disable=unused-argument
    ) -> Union[torch.Tensor, Sequence[Numeric]]:
        # pylint: disable-next=line-too-long
        """Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched."""
        raise NotImplementedError


Samplable.register(Distribution)


def _zero_order_naive(  # pylint: disable=too-many-statements
    fn: Callable[..., torch.Tensor],
    distribution: Samplable,
    argnums: Tuple[int, ...],
    num_samples: int,
    sigma: Numeric,
) -> Callable[..., torch.Tensor]:
    def apply(*args: Any) -> torch.Tensor:  # pylint: disable=too-many-statements
        diff_params = [args[argnum + 1] for argnum in argnums]
        flat_diff_params: List[Any]
        flat_diff_params, diff_params_treespec = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: List[Any]
                flat_args, args_treespec = pytree.tree_flatten(origin_args, none_is_leaf=True)  # type: ignore[arg-type]
                ctx.args_treespec = args_treespec

                is_tensor_mask = []
                tensors = []
                non_tensors = []
                for origin_arg in flat_args:
                    is_tensor = isinstance(origin_arg, torch.Tensor)
                    is_tensor_mask.append(is_tensor)
                    if is_tensor:
                        tensors.append(origin_arg)
                    else:
                        non_tensors.append(origin_arg)

                ctx.non_tensors = non_tensors
                ctx.is_tensor_mask = is_tensor_mask

                output = fn(*origin_args)
                ctx.distribution = distribution(*origin_args)
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError('`output` must be a tensor.')
                if output.ndim != 0:
                    raise RuntimeError('`output` must be a scalar tensor.')
                ctx.save_for_backward(*flat_diff_params, *tensors)
                ctx.len_args = len(args)
                ctx.len_params = len(flat_diff_params)
                return output

            @staticmethod
            def backward(  # pylint: disable=too-many-locals
                ctx: Any, *grad_outputs: Any
            ) -> TupleOfOptionalTensors:
                saved_tensors = ctx.saved_tensors
                flat_diff_params = saved_tensors[: ctx.len_params]
                tensors = saved_tensors[ctx.len_params :]
                non_tensors = ctx.non_tensors

                flat_args = []
                tensors_counter = 0
                non_tensors_counter = 0
                for is_tensor in ctx.is_tensor_mask:
                    if is_tensor:
                        flat_args.append(tensors[tensors_counter])
                        tensors_counter += 1
                    else:
                        flat_args.append(non_tensors[non_tensors_counter])
                        non_tensors_counter += 1

                args: List[Any] = pytree.tree_unflatten(ctx.args_treespec, flat_args)  # type: ignore[assignment]

                def add_perturbation(tensor, noises):
                    return tensor.add(noises, alpha=sigma)

                param_grads: ListOfTensors = [0.0 for _ in range(len(flat_diff_params))]  # type: ignore[misc]

                for _ in range(num_samples):
                    noises = [ctx.distribution(sample_shape=p.shape) for p in flat_diff_params]
                    flat_noisy_params = [
                        add_perturbation(t, n) for t, n in zip(flat_diff_params, noises)
                    ]
                    noisy_params: List[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                        diff_params_treespec, flat_noisy_params
                    )

                    for argnum, noisy_param in zip(argnums, noisy_params):
                        args[argnum] = noisy_param

                    output = fn(*args)
                    weighted_grad = grad_outputs[0].mul(output).mul_(1 / sigma)

                    for i, noise in enumerate(noises):
                        param_grads[i] += weighted_grad * noise

                for i in range(len(flat_diff_params)):
                    param_grads[i] /= num_samples

                return tuple(param_grads + [None] * (ctx.len_args - len(flat_diff_params)))

        return ZeroOrder.apply(*flat_diff_params, (args,))

    return apply


def _zero_order_forward(  # pylint: disable=too-many-statements
    fn: Callable[..., torch.Tensor],
    distribution: Samplable,
    argnums: Tuple[int, ...],
    num_samples: int,
    sigma: Numeric,
) -> Callable[..., torch.Tensor]:
    def apply(*args: Any) -> torch.Tensor:  # pylint: disable=too-many-statements
        diff_params = [args[argnum + 1] for argnum in argnums]
        flat_diff_params: List[Any]
        flat_diff_params, diff_params_treespec = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: List[Any]
                flat_args, args_treespec = pytree.tree_flatten(origin_args, none_is_leaf=True)  # type: ignore[arg-type]
                ctx.args_treespec = args_treespec

                is_tensor_mask = []
                tensors = []
                non_tensors = []
                for origin_arg in flat_args:
                    is_tensor = isinstance(origin_arg, torch.Tensor)
                    is_tensor_mask.append(is_tensor)
                    if is_tensor:
                        tensors.append(origin_arg)
                    else:
                        non_tensors.append(origin_arg)

                ctx.non_tensors = non_tensors
                ctx.is_tensor_mask = is_tensor_mask

                output = fn(*origin_args)
                ctx.distribution = distribution(*origin_args)
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError('`output` must be a tensor.')
                if output.ndim != 0:
                    raise RuntimeError('`output` must be a scalar tensor.')
                ctx.save_for_backward(*flat_diff_params, *tensors, output)
                ctx.len_args = len(args)
                ctx.len_params = len(flat_diff_params)
                return output

            @staticmethod
            def backward(  # pylint: disable=too-many-locals
                ctx: Any, *grad_outputs: Any
            ) -> TupleOfOptionalTensors:
                saved_tensors = ctx.saved_tensors
                flat_diff_params = saved_tensors[: ctx.len_params]
                tensors = saved_tensors[ctx.len_params : -1]
                output = saved_tensors[-1]
                non_tensors = ctx.non_tensors

                flat_args = []
                tensors_counter = 0
                non_tensors_counter = 0
                for is_tensor in ctx.is_tensor_mask:
                    if is_tensor:
                        flat_args.append(tensors[tensors_counter])
                        tensors_counter += 1
                    else:
                        flat_args.append(non_tensors[non_tensors_counter])
                        non_tensors_counter += 1

                args: List[Any] = pytree.tree_unflatten(ctx.args_treespec, flat_args)  # type: ignore[assignment]

                def add_perturbation(tensor, noises):
                    return tensor.add(noises, alpha=sigma)

                param_grads: ListOfTensors = [0.0 for _ in range(len(flat_diff_params))]  # type: ignore[misc]

                for _ in range(num_samples):
                    noises = [ctx.distribution(sample_shape=p.shape) for p in flat_diff_params]
                    flat_noisy_params = [
                        add_perturbation(t, n) for t, n in zip(flat_diff_params, noises)
                    ]
                    noisy_params: List[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                        diff_params_treespec, flat_noisy_params
                    )

                    for argnum, noisy_param in zip(argnums, noisy_params):
                        args[argnum] = noisy_param

                    noisy_output = fn(*args)
                    output = noisy_output - output
                    weighted_grad = grad_outputs[0].mul(output).div_(1.0 / sigma)

                    for i, noise in enumerate(noises):
                        param_grads[i] += weighted_grad * noise

                for i in range(len(flat_diff_params)):
                    param_grads[i] /= num_samples

                return tuple(param_grads + [None] * (ctx.len_args - len(flat_diff_params)))

        return ZeroOrder.apply(*flat_diff_params, (args,))

    return apply


def _zero_order_antithetic(  # pylint: disable=too-many-statements
    fn: Callable[..., torch.Tensor],
    distribution: Samplable,
    argnums: Tuple[int, ...],
    num_samples: int,
    sigma: Numeric,
) -> Callable[..., torch.Tensor]:
    def apply(*args: Any) -> torch.Tensor:  # pylint: disable=too-many-statements
        diff_params = [args[argnum + 1] for argnum in argnums]
        flat_diff_params: List[Any]
        flat_diff_params, diff_params_treespec = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: List[Any]
                flat_args, args_treespec = pytree.tree_flatten(origin_args, none_is_leaf=True)  # type: ignore[arg-type]
                ctx.args_treespec = args_treespec

                is_tensor_mask = []
                tensors = []
                non_tensors = []
                for origin_arg in flat_args:
                    is_tensor = isinstance(origin_arg, torch.Tensor)
                    is_tensor_mask.append(is_tensor)
                    if is_tensor:
                        tensors.append(origin_arg)
                    else:
                        non_tensors.append(origin_arg)

                ctx.non_tensors = non_tensors
                ctx.is_tensor_mask = is_tensor_mask

                output = fn(*origin_args)
                ctx.distribution = distribution(*origin_args)
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError('`output` must be a tensor.')
                if output.ndim != 0:
                    raise RuntimeError('`output` must be a scalar tensor.')
                ctx.save_for_backward(*flat_diff_params, *tensors)
                ctx.len_args = len(args)
                ctx.len_params = len(flat_diff_params)
                return output

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any):  # pylint: disable=too-many-locals
                saved_tensors = ctx.saved_tensors
                flat_diff_params = saved_tensors[: ctx.len_params]
                tensors = saved_tensors[ctx.len_params :]
                non_tensors = ctx.non_tensors

                flat_args = []
                tensors_counter = 0
                non_tensors_counter = 0
                for is_tensor in ctx.is_tensor_mask:
                    if is_tensor:
                        flat_args.append(tensors[tensors_counter])
                        tensors_counter += 1
                    else:
                        flat_args.append(non_tensors[non_tensors_counter])
                        non_tensors_counter += 1

                args: List[Any] = pytree.tree_unflatten(ctx.args_treespec, flat_args)  # type: ignore[assignment]

                param_grads: ListOfTensors = [0.0 for _ in range(len(flat_diff_params))]  # type: ignore[misc]

                def get_output(add_perturbation_fn, noises) -> torch.Tensor:
                    flat_noisy_params = [
                        add_perturbation_fn(t, n, alpha=sigma)
                        for t, n in zip(flat_diff_params, noises)
                    ]
                    noisy_params: List[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                        diff_params_treespec, flat_noisy_params
                    )

                    for argnum, noisy_param in zip(argnums, noisy_params):
                        args[argnum] = noisy_param

                    return fn(*args)

                for _ in range(num_samples):
                    noises = [ctx.distribution(sample_shape=p.shape) for p in flat_diff_params]
                    output = get_output(torch.add, noises) - get_output(torch.sub, noises)
                    weighted_grad = grad_outputs[0].mul(output).mul_(0.5 / sigma)

                    for i, noise in enumerate(noises):
                        param_grads[i] += weighted_grad * noise

                for i in range(len(flat_diff_params)):
                    param_grads[i] /= num_samples

                return tuple(param_grads + [None] * (ctx.len_args - len(flat_diff_params)))

        return ZeroOrder.apply(*flat_diff_params, (args,))

    return apply


Algorithm: TypeAlias = Literal['naive', 'forward', 'antithetic']


def zero_order(
    distribution: Samplable,
    algo: Algorithm = 'naive',
    argnums: Union[int, Tuple[int, ...]] = (0,),
    num_samples: int = 1,
    sigma: Numeric = 1.0,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Decorator for applying zero-order differentiation.

    Args:
        distribution: (object)
            A sampler object, it should have method ``sample(sample_shape)`` to sample perturbations
            from the given distribution.
        algo: (str)
            The algorithm to use. The currently supported algorithms are :const:`'naive'`,
            :const:`'forward'`, and :const:`'antithetic'`. Defaults to :const:`'naive'`.
        argnums: (int or tuple of int, default: :const:`0`)
            Specifies arguments to compute gradients with respect to.
        num_samples: (int, default :const:`1`)
            The number of sample to get the averaged estimated gradient.
        sigma: (Numeric)
            The standard deviation of the perturbation. Defaults to :const:`1.0`.

    Returns:
        A function decorator that enables zero-order gradient estimation.
    """
    assert algo in ('naive', 'forward', 'antithetic')
    if algo == 'naive':
        algo_fn = _zero_order_naive
    elif algo == 'forward':
        algo_fn = _zero_order_forward
    else:
        algo_fn = _zero_order_antithetic

    if isinstance(argnums, int):
        argnums = (argnums,)

    return functools.partial(
        algo_fn,
        distribution=distribution,
        argnums=argnums,
        num_samples=num_samples,
        sigma=sigma,
    )


class _CallableModule(_ModuleType):  # pylint: disable=too-few-public-methods
    def __call__(self, *args, **kwargs):
        return self.zero_order(*args, **kwargs)


# Replace entry in sys.modules for this module with an instance of _CallableModule
_modself = _sys.modules[__name__]
_modself.__class__ = _CallableModule
del _sys, _ModuleType, _modself, _CallableModule
