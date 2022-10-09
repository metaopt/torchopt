# Copyright 2022 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import functools
from typing import Any, Callable, Dict, Tuple, Union

import optree
import torch
from torch.autograd import Function


def _zero_order_a(
    fn: Callable,
    distribution: Any,
    argnums: Union[int, Tuple[int]],
    **kwargs: Dict[str, Any],
) -> Callable:
    sigma = kwargs['sigma']
    if isinstance(argnums, int):
        argnums = (argnums,)

    def apply(*args):
        diff_params = [args[argnum] for argnum in argnums]
        flatten_diff_params, diff_params_tree = optree.tree_flatten(diff_params)

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                flatten_diff_params = args[:-1]
                origin_args = list(args[-1][0])

                flatten_args, args_tree = optree.tree_flatten(origin_args)
                ctx.args_tree = args_tree

                tensor_mask = []
                tensors = []
                non_tensors = []
                for origin_arg in flatten_args:
                    is_tensor = isinstance(origin_arg, torch.Tensor)
                    tensor_mask.append(is_tensor)
                    if is_tensor:
                        tensors.append(origin_arg)
                    else:
                        non_tensors.append(origin_arg)

                ctx.non_tensors = non_tensors
                ctx.tensor_mask = tensor_mask

                ctx.save_for_backward(*flatten_diff_params, *tensors)
                ctx.len_args = len(args)
                ctx.len_params = len(flatten_diff_params)
                return fn(*origin_args)

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:  # pylint: disable=too-many-locals
                saved_tensors = ctx.saved_tensors
                flatten_diff_params = saved_tensors[: ctx.len_params]
                tensors = saved_tensors[ctx.len_params :]
                non_tensors = ctx.non_tensors

                args = []
                tensors_counter = 0
                non_tensors_counter = 0
                for is_tensor in ctx.tensor_mask:
                    if is_tensor:
                        args.append(tensors[tensors_counter])
                        tensors_counter += 1
                    else:
                        args.append(non_tensors[non_tensors_counter])
                        non_tensors_counter += 1

                args = ctx.args_tree.unflatten(args)
                args = list(args)

                def add_perturbation(tensor, noise):
                    return tensor.add(noise, alpha=sigma)

                noise = [distribution.sample(sample_shape=p.shape) for p in flatten_diff_params]
                flatten_noisy_params = [
                    add_perturbation(t, n) for t, n in zip(flatten_diff_params, noise)
                ]
                noisy_params = diff_params_tree.unflatten(flatten_noisy_params)

                for argnum, noisy_param in zip(argnums, noisy_params):
                    args[argnum] = noisy_param

                loss = fn(*args)
                weighted_grad = grad_outputs[0].mul(loss).mul_(1 / sigma)

                out_grad = [None for _ in range(ctx.len_args)]
                for i in range(len(flatten_diff_params)):
                    out_grad[i] = weighted_grad * noise[i]
                return tuple(out_grad)

        return ZeroOrder.apply(*flatten_diff_params, (args,))

    return apply


def _zero_order_b(
    fn: Callable,
    distribution: Any,
    argnums: Union[int, Tuple[int]],
    **kwargs: Dict[str, Any],
) -> Callable:
    sigma = kwargs['sigma']
    if isinstance(argnums, int):
        argnums = (argnums,)

    def apply(*args):
        diff_params = [args[argnum] for argnum in argnums]
        flatten_diff_params, diff_params_tree = optree.tree_flatten(diff_params)

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(
                ctx: Any, *args: Any, **kwargs: Any
            ) -> Any:  # pylint: disable=arguments-differ
                flatten_diff_params = args[:-1]
                origin_args = list(args[-1][0])

                flatten_args, args_tree = optree.tree_flatten(origin_args)
                ctx.args_tree = args_tree

                tensor_mask = []
                tensors = []
                non_tensors = []
                for origin_arg in flatten_args:
                    is_tensor = isinstance(origin_arg, torch.Tensor)
                    tensor_mask.append(is_tensor)
                    if is_tensor:
                        tensors.append(origin_arg)
                    else:
                        non_tensors.append(origin_arg)

                ctx.non_tensors = non_tensors
                ctx.tensor_mask = tensor_mask

                loss = fn(*origin_args)
                ctx.save_for_backward(*flatten_diff_params, *tensors, loss)
                ctx.len_args = len(args)
                ctx.len_params = len(flatten_diff_params)
                return loss

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:  # pylint: disable=too-many-locals
                saved_tensors = ctx.saved_tensors
                flatten_diff_params = saved_tensors[: ctx.len_params]
                tensors = saved_tensors[ctx.len_params : -1]
                loss = saved_tensors[-1]
                non_tensors = ctx.non_tensors

                args = []
                tensors_counter = 0
                non_tensors_counter = 0
                for is_tensor in ctx.tensor_mask:
                    if is_tensor:
                        args.append(tensors[tensors_counter])
                        tensors_counter += 1
                    else:
                        args.append(non_tensors[non_tensors_counter])
                        non_tensors_counter += 1

                args = ctx.args_tree.unflatten(args)
                args = list(args)

                def add_perturbation(tensor, noise):
                    return tensor.add(noise, alpha=sigma)

                noise = [distribution.sample(sample_shape=p.shape) for p in flatten_diff_params]
                flatten_noisy_params = [
                    add_perturbation(t, n) for t, n in zip(flatten_diff_params, noise)
                ]
                noisy_params = diff_params_tree.unflatten(flatten_noisy_params)

                for argnum, noisy_param in zip(argnums, noisy_params):
                    args[argnum] = noisy_param

                noisy_loss = fn(*args)
                loss = noisy_loss - loss
                weighted_grad = grad_outputs[0].mul(loss).mul_(1 / sigma)

                out_grad = [None for _ in range(ctx.len_args)]
                for i in range(len(flatten_diff_params)):
                    out_grad[i] = weighted_grad * noise[i]
                return tuple(out_grad)

        return ZeroOrder.apply(*flatten_diff_params, (args,))

    return apply


def _zero_order_c(
    fn: Callable,
    distribution: Any,
    argnums: Union[int, Tuple[int]],
    **kwargs: Dict[str, Any],
) -> Callable:
    sigma = kwargs['sigma']
    if isinstance(argnums, int):
        argnums = (argnums,)

    def apply(*args):
        diff_params = [args[argnum] for argnum in argnums]
        flatten_diff_params, diff_params_tree = optree.tree_flatten(diff_params)

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                flatten_diff_params = args[:-1]
                origin_args = list(args[-1][0])

                flatten_args, args_tree = optree.tree_flatten(origin_args)
                ctx.args_tree = args_tree

                tensor_mask = []
                tensors = []
                non_tensors = []
                for origin_arg in flatten_args:
                    is_tensor = isinstance(origin_arg, torch.Tensor)
                    tensor_mask.append(is_tensor)
                    if is_tensor:
                        tensors.append(origin_arg)
                    else:
                        non_tensors.append(origin_arg)

                ctx.non_tensors = non_tensors
                ctx.tensor_mask = tensor_mask

                ctx.save_for_backward(*flatten_diff_params, *tensors)
                ctx.len_args = len(args)
                ctx.len_params = len(flatten_diff_params)
                return fn(*origin_args)

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:  # pylint: disable=too-many-locals
                saved_tensors = ctx.saved_tensors
                flatten_diff_params = saved_tensors[: ctx.len_params]
                tensors = saved_tensors[ctx.len_params :]
                non_tensors = ctx.non_tensors

                args = []
                tensors_counter = 0
                non_tensors_counter = 0
                for is_tensor in ctx.tensor_mask:
                    if is_tensor:
                        args.append(tensors[tensors_counter])
                        tensors_counter += 1
                    else:
                        args.append(non_tensors[non_tensors_counter])
                        non_tensors_counter += 1

                args = ctx.args_tree.unflatten(args)
                args = list(args)

                noise = [distribution.sample(sample_shape=p.shape) for p in flatten_diff_params]

                def get_loss(transform_fn):
                    flatten_noisy_params = [
                        transform_fn(t, n) for t, n in zip(flatten_diff_params, noise)
                    ]
                    noisy_params = diff_params_tree.unflatten(flatten_noisy_params)

                    for argnum, noisy_param in zip(argnums, noisy_params):
                        args[argnum] = noisy_param

                    return fn(*args)

                loss = get_loss(lambda tensor, noise: torch.add(noise, alpha=sigma)) - get_loss(
                    lambda tensor, noise: torch.sub(noise, alpha=sigma)
                )
                weighted_grad = grad_outputs[0].mul(loss).mul_(0.5 / sigma)

                out_grad = [None for _ in range(ctx.len_args)]
                for i in range(len(flatten_diff_params)):
                    out_grad[i] = weighted_grad * noise[i]
                return tuple(out_grad)

        return ZeroOrder.apply(*flatten_diff_params, (args,))

    return apply


def zero_order(
    distribution: Any,
    algo: str,
    argnums: Union[int, Tuple[int]] = (0,),
    **kwargs,
):
    """Decorator for applying zero order differentiation.

    Args:
        distribution: (object)
            A sampler object, it should have method ``sample(sample_shape)`` to sample
            perturbations from the given distribution.
        algo: (str)
            The algorithm to use.
        argnums: (int or tuple of int, default: :const:`0`)
            Specifies arguments to compute gradients with respect to.

    Returns:
        A zero order function decorator.
    """
    return functools.partial(
        _zero_order_a, distribution=distribution, algo=algo, argnums=argnums, **kwargs
    )
