# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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

from __future__ import annotations

import functools
from typing import Any, Callable, Literal, Sequence
from typing_extensions import TypeAlias  # Python 3.10+

import torch
from torch.autograd import Function

from torchopt import pytree
from torchopt.typing import ListOfTensors, Numeric, Samplable, SampleFunc, TupleOfOptionalTensors


class WrappedSamplable(Samplable):  # pylint: disable=too-few-public-methods
    """A wrapper that wraps a sample function to a :class:`Samplable` object."""

    def __init__(self, sample_fn: SampleFunc) -> None:
        """Wrap a sample function to make it a :class:`Samplable` object."""
        self.sample_fn = sample_fn

    def sample(
        self,
        sample_shape: torch.Size = torch.Size(),  # noqa: B008
    ) -> torch.Tensor | Sequence[Numeric]:
        # pylint: disable-next=line-too-long
        """Generate a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched."""
        return self.sample_fn(sample_shape)


def _zero_order_naive(  # pylint: disable=too-many-statements
    fn: Callable[..., torch.Tensor],
    distribution: Samplable,
    argnums: tuple[int, ...],
    num_samples: int,
    sigma: float,
) -> Callable[..., torch.Tensor]:
    @functools.wraps(fn)
    def apply(*args: Any) -> torch.Tensor:  # pylint: disable=too-many-statements
        diff_params = [args[argnum] for argnum in argnums]
        flat_diff_params: list[Any]
        flat_diff_params, diff_params_treespec = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: list[Any]
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
                ctx: Any,
                *grad_outputs: Any,
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

                args: list[Any] = pytree.tree_unflatten(ctx.args_treespec, flat_args)  # type: ignore[assignment]

                def add_perturbation(
                    tensor: torch.Tensor,
                    noise: torch.Tensor | Numeric,
                ) -> torch.Tensor:
                    return tensor.add(noise, alpha=sigma)

                param_grads: ListOfTensors = [0.0 for _ in range(len(flat_diff_params))]  # type: ignore[misc]

                for _ in range(num_samples):
                    noises = [distribution.sample(sample_shape=p.shape) for p in flat_diff_params]
                    flat_noisy_params = [
                        add_perturbation(t, n) for t, n in zip(flat_diff_params, noises)  # type: ignore[arg-type]
                    ]
                    noisy_params: list[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                        diff_params_treespec,
                        flat_noisy_params,
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
    argnums: tuple[int, ...],
    num_samples: int,
    sigma: float,
) -> Callable[..., torch.Tensor]:
    @functools.wraps(fn)
    def apply(*args: Any) -> torch.Tensor:  # pylint: disable=too-many-statements
        diff_params = [args[argnum] for argnum in argnums]
        flat_diff_params: list[Any]
        flat_diff_params, diff_params_treespec = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: list[Any]
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
                ctx: Any,
                *grad_outputs: Any,
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

                args: list[Any] = pytree.tree_unflatten(ctx.args_treespec, flat_args)  # type: ignore[assignment]

                def add_perturbation(tensor: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
                    return tensor.add(noise, alpha=sigma)

                param_grads: ListOfTensors = [0.0 for _ in range(len(flat_diff_params))]  # type: ignore[misc]

                for _ in range(num_samples):
                    noises = [distribution.sample(sample_shape=p.shape) for p in flat_diff_params]
                    flat_noisy_params = [
                        add_perturbation(t, n) for t, n in zip(flat_diff_params, noises)  # type: ignore[arg-type]
                    ]
                    noisy_params: list[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                        diff_params_treespec,
                        flat_noisy_params,
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
    argnums: tuple[int, ...],
    num_samples: int,
    sigma: float,
) -> Callable[..., torch.Tensor]:
    @functools.wraps(fn)
    def apply(*args: Any) -> torch.Tensor:  # pylint: disable=too-many-statements
        diff_params = [args[argnum] for argnum in argnums]
        flat_diff_params: list[Any]
        flat_diff_params, diff_params_treespec = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: list[Any]
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
                ctx: Any,
                *grad_outputs: Any,
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

                args: list[Any] = pytree.tree_unflatten(ctx.args_treespec, flat_args)  # type: ignore[assignment]

                param_grads: ListOfTensors = [0.0 for _ in range(len(flat_diff_params))]  # type: ignore[misc]

                def get_output(
                    add_perturbation_fn: Callable,
                    noises: Sequence[torch.Tensor | Numeric],
                ) -> torch.Tensor:
                    flat_noisy_params = [
                        add_perturbation_fn(t, n, alpha=sigma)
                        for t, n in zip(flat_diff_params, noises)
                    ]
                    noisy_params: list[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                        diff_params_treespec,
                        flat_noisy_params,
                    )

                    for argnum, noisy_param in zip(argnums, noisy_params):
                        args[argnum] = noisy_param

                    return fn(*args)

                for _ in range(num_samples):
                    noises = [distribution.sample(sample_shape=p.shape) for p in flat_diff_params]
                    output = get_output(torch.add, noises) - get_output(torch.sub, noises)  # type: ignore[arg-type]
                    weighted_grad = grad_outputs[0].mul(output).mul_(0.5 / sigma)

                    for i, noise in enumerate(noises):
                        param_grads[i] += weighted_grad * noise

                for i in range(len(flat_diff_params)):
                    param_grads[i] /= num_samples

                return tuple(param_grads + [None] * (ctx.len_args - len(flat_diff_params)))

        return ZeroOrder.apply(*flat_diff_params, (args,))

    return apply


Method: TypeAlias = Literal['naive', 'forward', 'antithetic']


def zero_order(
    distribution: SampleFunc | Samplable,
    method: Method = 'naive',
    argnums: int | tuple[int, ...] = (0,),
    num_samples: int = 1,
    sigma: float = 1.0,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Return a decorator for applying zero-order differentiation.

    Args:
        distribution (callable or Samplable): A samplable object that has method
            ``samplable.sample(sample_shape)`` or a function that takes the shape as input and
            returns a shaped batch of samples. This is used to sample perturbations from the given
            distribution. The distribution should be sphere symmetric.
        method (str, optional): The algorithm to use. The currently supported algorithms are
            :const:`'naive'`, :const:`'forward'`, and :const:`'antithetic'`.
            (default: :const:`'naive'`)
        argnums (int or tuple of int, optional): Specifies arguments to compute gradients with
            respect to. (default: :const:`0`)
        num_samples (int, optional): The number of sample to get the averaged estimated gradient.
            (default: :const:`1`)
        sigma (float, optional): The standard deviation of the perturbation.
            (default: :const:`1.0`)

    Returns:
        A function decorator that enables zero-order gradient estimation.
    """
    assert method in ('naive', 'forward', 'antithetic')
    if method == 'naive':
        method_fn = _zero_order_naive
    elif method == 'forward':
        method_fn = _zero_order_forward
    else:
        method_fn = _zero_order_antithetic

    if isinstance(argnums, int):
        argnums = (argnums,)

    if not isinstance(distribution, Samplable):
        if not callable(distribution):
            raise TypeError('`distribution` must be a callable or an instance of `Samplable`.')
        distribution = WrappedSamplable(distribution)

    return functools.partial(
        method_fn,
        distribution=distribution,
        argnums=argnums,
        num_samples=num_samples,
        sigma=sigma,
    )
