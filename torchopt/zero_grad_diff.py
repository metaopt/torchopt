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
"""Zero-Order Gradient Estimation."""

import functools
from typing import Any, Callable, List, Sequence, Tuple, Union
from typing_extensions import Literal  # Python 3.8+
from typing_extensions import Protocol  # Python 3.8+
from typing_extensions import TypeAlias  # Python 3.10+

import torch
from torch.autograd import Function
from torch.distributions import Distribution

from torchopt import pytree
from torchopt.typing import Numeric


class Samplable(Protocol):  # pylint: disable=too-few-public-methods
    """Abstract protocol class that supports sampling."""

    def sample(
        self, sample_shape: torch.Size = torch.Size()  # pylint: disable=unused-argument
    ) -> Union[torch.Tensor, Sequence[Numeric]]:
        # pylint: disable-next=line-too-long
        """Generates a sample_shape shaped sample or sample_shape shaped batch of samples if the distribution parameters are batched."""
        raise NotImplementedError


Samplable.register(Distribution)


def _zero_order_naive(
    fn: Callable[..., torch.Tensor],
    distribution: Samplable,
    argnums: Tuple[int, ...],
    sigma: Numeric,
) -> Callable[..., torch.Tensor]:
    def apply(*args: Any) -> torch.Tensor:
        diff_params = [args[argnum] for argnum in argnums]
        flat_diff_params: List[Any]
        flat_diff_params, diff_params_treedef = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx, *args, **kwargs):
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: List[Any]
                flat_args, args_treedef = pytree.tree_flatten(origin_args)  # type: ignore[arg-type]
                ctx.args_treedef = args_treedef

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

                ctx.save_for_backward(*flat_diff_params, *tensors)
                ctx.len_args = len(args)
                ctx.len_params = len(flat_diff_params)
                return fn(*origin_args)

            @staticmethod
            def backward(ctx, *grad_outputs):  # pylint: disable=too-many-locals
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

                args: List[Any] = pytree.tree_unflatten(ctx.args_treedef, flat_args)  # type: ignore[assignment]

                def add_perturbation(tensor, noise):
                    return tensor.add(noise, alpha=sigma)

                noise = [distribution.sample(sample_shape=p.shape) for p in flat_diff_params]
                flat_noisy_params = [
                    add_perturbation(t, n) for t, n in zip(flat_diff_params, noise)
                ]
                noisy_params: List[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                    diff_params_treedef, flat_noisy_params
                )

                for argnum, noisy_param in zip(argnums, noisy_params):
                    args[argnum] = noisy_param

                loss = fn(*args)
                weighted_grad = grad_outputs[0].mul(loss).mul_(1 / sigma)

                out_grads = [None for _ in range(ctx.len_args)]
                for i in range(len(flat_diff_params)):
                    out_grads[i] = weighted_grad * noise[i]
                return tuple(out_grads)

        return ZeroOrder.apply(*flat_diff_params, (args,))

    return apply


def _zero_order_forward(
    fn: Callable[..., torch.Tensor],
    distribution: Samplable,
    argnums: Tuple[int, ...],
    sigma: Numeric,
) -> Callable[..., torch.Tensor]:
    def apply(*args: Any) -> torch.Tensor:
        diff_params = [args[argnum] for argnum in argnums]
        flat_diff_params: List[Any]
        flat_diff_params, diff_params_treedef = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(
                ctx: Any, *args: Any, **kwargs: Any
            ) -> Any:  # pylint: disable=arguments-differ
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: List[Any]
                flat_args, args_treedef = pytree.tree_flatten(origin_args)  # type: ignore[arg-type]
                ctx.args_treedef = args_treedef

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

                loss = fn(*origin_args)
                ctx.save_for_backward(*flat_diff_params, *tensors, loss)
                ctx.len_args = len(args)
                ctx.len_params = len(flat_diff_params)
                return loss

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:  # pylint: disable=too-many-locals
                saved_tensors = ctx.saved_tensors
                flat_diff_params = saved_tensors[: ctx.len_params]
                tensors = saved_tensors[ctx.len_params : -1]
                loss = saved_tensors[-1]
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

                args: List[Any] = pytree.tree_unflatten(ctx.args_treedef, flat_args)  # type: ignore[assignment]

                def add_perturbation(tensor, noise):
                    return tensor.add(noise, alpha=sigma)

                noise = [distribution.sample(sample_shape=p.shape) for p in flat_diff_params]
                flat_noisy_params = [
                    add_perturbation(t, n) for t, n in zip(flat_diff_params, noise)
                ]
                noisy_params: List[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                    diff_params_treedef, flat_noisy_params
                )

                for argnum, noisy_param in zip(argnums, noisy_params):
                    args[argnum] = noisy_param

                noisy_loss = fn(*args)
                loss = noisy_loss - loss
                weighted_grad = grad_outputs[0].mul(loss).mul_(1 / sigma)

                out_grads = [None for _ in range(ctx.len_args)]
                for i in range(len(flat_diff_params)):
                    out_grads[i] = weighted_grad * noise[i]
                return tuple(out_grads)

        return ZeroOrder.apply(*flat_diff_params, (args,))

    return apply


def _zero_order_antithetic(
    fn: Callable[..., torch.Tensor],
    distribution: Samplable,
    argnums: Tuple[int, ...],
    sigma: Numeric,
) -> Callable[..., torch.Tensor]:
    def apply(*args: Any) -> torch.Tensor:
        diff_params = [args[argnum] for argnum in argnums]
        flat_diff_params: List[Any]
        flat_diff_params, diff_params_treedef = pytree.tree_flatten(diff_params)  # type: ignore[arg-type]

        class ZeroOrder(Function):  # pylint: disable=missing-class-docstring,abstract-method
            @staticmethod
            def forward(ctx, *args, **kwargs):
                flat_diff_params = args[:-1]
                origin_args = list(args[-1][0])
                flat_args: List[Any]
                flat_args, args_treedef = pytree.tree_flatten(origin_args)  # type: ignore[arg-type]
                ctx.args_treedef = args_treedef

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

                ctx.save_for_backward(*flat_diff_params, *tensors)
                ctx.len_args = len(args)
                ctx.len_params = len(flat_diff_params)
                return fn(*origin_args)

            @staticmethod
            def backward(ctx, *grad_outputs):  # pylint: disable=too-many-locals
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

                args: List[Any] = pytree.tree_unflatten(ctx.args_treedef, flat_args)  # type: ignore[assignment]

                noise = [distribution.sample(sample_shape=p.shape) for p in flat_diff_params]

                def get_loss(add_perturbation_fn) -> torch.Tensor:
                    flat_noisy_params = [
                        add_perturbation_fn(t, n, alpha=sigma)
                        for t, n in zip(flat_diff_params, noise)
                    ]
                    noisy_params: List[Any] = pytree.tree_unflatten(  # type: ignore[assignment]
                        diff_params_treedef, flat_noisy_params
                    )

                    for argnum, noisy_param in zip(argnums, noisy_params):
                        args[argnum] = noisy_param

                    return fn(*args)

                loss = get_loss(torch.add) - get_loss(torch.sub)
                weighted_grad = grad_outputs[0].mul(loss).mul_(0.5 / sigma)

                out_grads = [None for _ in range(ctx.len_args)]
                for i in range(len(flat_diff_params)):
                    out_grads[i] = weighted_grad * noise[i]
                return tuple(out_grads)

        return ZeroOrder.apply(*flat_diff_params, (args,))

    return apply


Algorithm: TypeAlias = Literal['naive', 'forward', 'antithetic']


def zero_order(
    distribution: Samplable,
    algo: Algorithm = 'naive',
    argnums: Union[int, Tuple[int, ...]] = (0,),
    sigma: Numeric = 1.0,
) -> Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]:
    """Decorator for applying zero order differentiation.

    Args:
        distribution: (object)
            A sampler object, it should have method ``sample(sample_shape)`` to sample perturbations
            from the given distribution.
        algo: (str)
            The algorithm to use. The currently supported algorithms are :const:`'naive'`,
            :const:`'forward'`, and :const:`'antithetic'`. Defaults to :const:`'naive'`.
        argnums: (int or tuple of int, default: :const:`0`)
            Specifies arguments to compute gradients with respect to.
        sigma: (Numeric)
            The standard deviation of the perturbation. Defaults to :const:`1.0`.

    Returns:
        A zero order function decorator.
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

    return functools.partial(algo_fn, distribution=distribution, argnums=argnums, sigma=sigma)
