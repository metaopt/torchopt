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
"""Implicit Meta-Gradient."""

# pylint: disable=invalid-name

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import functorch
import torch
from torch.autograd import Function

from torchopt import linear_solve, pytree


__all__ = ['custom_root']


Args = Tuple[Any, ...]
KwArgs = Dict[str, Any]


class MaskedOptimalityFn:  # pylint: disable=missing-class-docstring,too-few-public-methods
    def __init__(
        self,
        optimality_fn: Callable,
        solution: Any,
        result_is_tensor: bool,
        argnums: Tuple[int, ...],
        *args,
    ) -> None:
        self.optimality_fn = optimality_fn
        self.solution = solution
        self.result_is_tensor = result_is_tensor
        self.argnums = argnums

        pre_filled = []
        post_filled = []
        for idx, arg in enumerate(args):
            if idx + 1 in argnums:  # plus 1 because we exclude the first argument
                post_filled.append(arg)
            else:
                pre_filled.append(arg)
        self.len_args = len(pre_filled) + len(post_filled)
        self.pre_filled = tuple(pre_filled)
        self.post_filled = tuple(post_filled)

    def __call__(self, *args) -> Any:
        true_args = []
        pre_filled_counter = 0
        for idx in range(self.len_args):
            if idx + 1 in self.argnums:  # plus 1 because we exclude the first argument
                arg = args[idx]
            else:
                arg = self.pre_filled[pre_filled_counter]
                pre_filled_counter += 1
            true_args.append(arg)
        if self.result_is_tensor:
            return self.optimality_fn(self.solution[0], *true_args)
        return self.optimality_fn(self.solution, *true_args)


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches
def _root_vjp(
    optimality_fn: Callable,
    solution: Any,
    args: Args,
    grad_outputs: Any,
    result_is_tensor: bool,
    argnums: Tuple[int, ...],
    solve: Callable = linear_solve.solve_normal_cg(),
) -> Tuple[Any, ...]:

    if result_is_tensor:

        def optimality_cond(solution):
            return optimality_fn(solution[0], *args)

    else:

        def optimality_cond(solution):
            return optimality_fn(solution, *args)

    _, vjp_optimality_cond, *_ = functorch.vjp(optimality_cond, solution)

    # Compute the multiplication A^T u = (u^T A)^T.
    if result_is_tensor:

        def matvec(u):
            return vjp_optimality_cond(u[0])[0]

    else:

        def matvec(u):
            return vjp_optimality_cond(u)[0]

    # The solution of A^T u = v, where
    # A = jacobian(optimality_fn, argnums=0)
    # v = -grad_outputs.
    v = pytree.tree_map(torch.neg, grad_outputs)
    u = solve(matvec, v)

    masked_optimality_fn = MaskedOptimalityFn(
        optimality_fn, solution, result_is_tensor, argnums, *args
    )

    if getattr(solve, 'is_sdp', False):
        if result_is_tensor:
            result = u[0]
        else:
            result = u
    else:
        _, vjp_optimality_fn, *_ = functorch.vjp(
            masked_optimality_fn, *masked_optimality_fn.post_filled
        )

        if result_is_tensor:
            result = vjp_optimality_fn(u[0])
        else:
            result = vjp_optimality_fn(u)

    true_result = [None]
    for idx in range(masked_optimality_fn.len_args):
        if idx + 1 in argnums:  # plus 1 because we exclude the first argument
            true_result.append(result[idx])
        else:
            true_result.append(None)

    return tuple(true_result)


def _extract_kwargs(kwarg_keys: Sequence[str], flat_args: Tuple[Any, ...]) -> Tuple[Args, KwArgs]:
    nargs = len(flat_args) - len(kwarg_keys)
    args, kwarg_vals = flat_args[:nargs], flat_args[nargs:]
    kwargs = dict(zip(kwarg_keys, kwarg_vals))
    return args, kwargs


def _signature_bind(signature: inspect.Signature, *args, **kwargs) -> Tuple[Args, KwArgs]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.args, bound.kwargs


def _signature_bind_and_match(
    signature: inspect.Signature, *args, **kwargs
) -> Tuple[Args, KwArgs, Callable[[Args], Tuple[Args, KwArgs]]]:
    # We want to bind *args and **kwargs based on the provided signature, but also to associate the
    # resulting positional arguments back. To achieve this, we lift arguments to a triple:
    #
    #   (was_kwarg, ref, value)
    #
    # where ref is an index position (int) if the original argument was from *args and a dictionary
    # key if the original argument was from **kwargs. After binding to the inspected signature, we
    # use the tags to associate the resolved positional arguments back to their arg and kwarg
    # source.

    args = [(False, i, v) for i, v in enumerate(args)]
    kwargs = {k: (True, k, v) for (k, v) in kwargs.items()}
    bound = signature.bind(*args, **kwargs)

    mapping = [(was_kwarg, ref) for was_kwarg, ref, _ in bound.args]

    def map_args_back(out_args):
        src_args = [None] * len(args)
        src_kwargs = {}
        for (was_kwarg, ref), out_arg in zip(mapping, out_args):
            if was_kwarg:
                src_kwargs[ref] = out_arg
            else:
                src_args[ref] = out_arg
        return src_args, src_kwargs

    out_args = tuple(v for _, _, v in bound.args)
    out_kwargs = {k: v for k, (_, _, v) in bound.kwargs.items()}
    return out_args, out_kwargs, map_args_back


def _split_tensor_and_others(
    mixed_tuple: Tuple[Any, ...],
) -> Tuple[pytree.PyTreeSpec, Tuple[bool, ...], Tuple[torch.Tensor, ...], Tuple[Any, ...]]:
    flattened: List[Any]
    flattened, treespec = pytree.tree_flatten(mixed_tuple, none_is_leaf=True)  # type: ignore[arg-type]
    tensors: List[torch.Tensor] = []
    non_tensors: List[Any] = []
    is_tensor_mask: List[bool] = []
    for item in flattened:
        is_tensor = isinstance(item, torch.Tensor)
        is_tensor_mask.append(is_tensor)
        if is_tensor:
            tensors.append(item.data)
        else:
            non_tensors.append(item)
    return treespec, tuple(is_tensor_mask), tuple(tensors), tuple(non_tensors)


def _merge_tensor_and_others(
    treespec: pytree.PyTreeSpec,
    is_tensor_mask: Tuple[bool, ...],
    tensors: Tuple[torch.Tensor, ...],
    non_tensors: Tuple[Any, ...],
) -> Any:
    tensor_counter = 0
    non_tensor_counter = 0
    results = []
    for is_tensor in is_tensor_mask:
        if is_tensor:
            results.append(tensors[tensor_counter])
            tensor_counter += 1
        else:
            results.append(non_tensors[non_tensor_counter])
            non_tensor_counter += 1
    return pytree.tree_unflatten(treespec, results)


# pylint: disable-next=too-many-arguments,too-many-statements
def _custom_root(
    solver_fn: Callable,
    optimality_fn: Callable,
    solve: Callable,
    argnums: Tuple[int, ...],
    has_aux: bool,
    reference_signature: Optional[Union[inspect.Signature, Callable]] = None,
) -> Callable:
    solver_fn_signature = inspect.signature(solver_fn)

    if reference_signature is None:
        reference_signature = inspect.signature(optimality_fn)
    elif not isinstance(reference_signature, inspect.Signature):
        # If is a CompositeLinearFunction, accesses subfn.
        # Otherwise, assumes a Callable.
        fn = getattr(reference_signature, 'subfn', reference_signature)
        reference_signature = inspect.signature(fn)

    def make_custom_vjp_solver_fn(solver_fn, kwarg_keys, args_sign):
        # pylint: disable-next=missing-class-docstring,abstract-method
        class ImplicitMetaGradient(Function):
            @staticmethod
            def forward(ctx, *flat_args):  # pylint: disable=arguments-differ
                args = []
                for idx, (start_point, is_tuple) in enumerate(args_sign):
                    if is_tuple:
                        args.append(tuple(flat_args[start_point : args_sign[idx + 1][0]]))
                    else:
                        args.append(flat_args[start_point])
                args = tuple(args)

                args, kwargs = _extract_kwargs(kwarg_keys, args)
                res = solver_fn(*args, **kwargs)
                (
                    args_treespec,
                    args_is_tensor_mask,
                    args_tensors,
                    args_non_tensors,
                ) = _split_tensor_and_others(args)
                ctx.args_treespec = args_treespec
                ctx.args_is_tensor_mask = args_is_tensor_mask
                ctx.args_non_tensors = args_non_tensors
                if has_aux:
                    res, aux = res
                    if torch.is_tensor(res):
                        ctx.save_for_backward(res, *args_tensors)
                        ctx.result_is_tensor = True
                        return (res, aux, True, torch.tensor)

                    ctx.save_for_backward(*res, *args_tensors)
                    ctx.result_is_tensor = False
                    return (*res, aux, False, type(res))

                if isinstance(res, torch.Tensor):
                    ctx.save_for_backward(res, *args_tensors)
                else:
                    ctx.save_for_backward(*res, *args_tensors)
                ctx.result_is_tensor = isinstance(res, torch.Tensor)
                return res

            @staticmethod
            def backward(ctx, *grad_outputs):  # pylint: disable=too-many-locals
                if has_aux:
                    grad_outputs = grad_outputs[:-3]

                saved_tensors = ctx.saved_tensors
                res, args_tensors = (
                    saved_tensors[: len(grad_outputs)],
                    saved_tensors[len(grad_outputs) :],
                )
                args_treespec = ctx.args_treespec
                args_is_tensor_mask = ctx.args_is_tensor_mask
                args_non_tensors = ctx.args_non_tensors
                args = _merge_tensor_and_others(
                    args_treespec, args_is_tensor_mask, args_tensors, args_non_tensors
                )

                args, kwargs = _extract_kwargs(kwarg_keys, args)

                solution = res
                bound_args, bound_kwargs, map_args_back = _signature_bind_and_match(
                    reference_signature, *args, **kwargs  # type: ignore[arg-type]
                )
                if bound_kwargs:
                    raise TypeError(
                        f'keyword arguments to solver_fn could not be resolved to positional '
                        f'arguments based on the signature {reference_signature}. This can '
                        f'happen under custom_root if optimality_fn takes catch-all **kwargs, or '
                        f'under custom_fixed_point if fixed_point_fn takes catch-all **kwargs, '
                        f'both of which are currently unsupported.'
                    )

                # Compute VJPs w.r.t. args.
                vjps = _root_vjp(
                    optimality_fn=optimality_fn,
                    solution=solution,
                    args=bound_args[1:],
                    grad_outputs=grad_outputs,
                    result_is_tensor=ctx.result_is_tensor,
                    argnums=argnums,
                    solve=solve,
                )
                # Prepend None as the vjp for init_params.

                args_vjps, kwargs_vjps = map_args_back(vjps)
                ordered_vjps = tuple(args_vjps) + tuple(kwargs_vjps[k] for k in kwargs.keys())
                true_vjps = []
                for (_, is_tuple), vjp in zip(args_sign, ordered_vjps):
                    if is_tuple:
                        for item in vjp:
                            true_vjps.append(item)
                    else:
                        true_vjps.append(vjp)
                return tuple(true_vjps)

        return ImplicitMetaGradient

    def wrapped_solver_fn(*args, **kwargs):
        args, kwargs = _signature_bind(solver_fn_signature, *args, **kwargs)
        keys, vals = list(kwargs.keys()), list(kwargs.values())

        args_sign = []
        flat_args = []
        args_counter = 0
        for idx, arg in enumerate(args):
            if idx in argnums:
                if isinstance(arg, torch.Tensor):
                    args_sign.append((args_counter, False))  # start position, is_tuple
                    flat_args.append(arg)
                    args_counter += 1
                elif isinstance(arg, tuple):
                    args_sign.append((args_counter, True))  # start position, is_tuple
                    for arg_item in arg:
                        flat_args.append(arg_item)
                    args_counter += len(arg)
                else:
                    raise RuntimeError('must be tensor or tensor tuple')
            else:
                args_sign.append((args_counter, False))  # start position, is_tuple
                flat_args.append(arg)
                args_counter += 1

        args_sign = tuple(args_sign)
        flat_args = tuple(flat_args)

        result = make_custom_vjp_solver_fn(solver_fn, keys, args_sign).apply(*flat_args, *vals)
        if has_aux:
            *res, aux, result_is_tensor, res_type = result
            if result_is_tensor:
                return res[0], aux
            res = res_type(res)
            return res, aux
        return result

    return wrapped_solver_fn


def custom_root(
    optimality_fn: Callable,
    argnums: Union[int, Tuple[int, ...]] = 0,
    has_aux: bool = False,
    solve: Callable = linear_solve.solve_normal_cg(),
) -> Callable[[Callable], Callable]:
    """Decorator for adding implicit differentiation to a root solver.

    This wrapper should be used as a decorator:

    .. code-block:: python

        def optimality_fn(optimal_params, ...):
            ...
            return residual

        @custom_root(optimality_fn, argnums=argnums)
        def solver_fn(params, arg1, arg2, ...):
            ...
            return optimal_params

    The first argument to ``optimality_fn`` and ``solver_fn`` is preserved as the parameter input.
    The ``argnums`` argument refers to the indices of the variables in ``solver_fn``'s signature.
    For example, setting ``argnums=(1, 2)`` will compute the gradient of ``optimal_params`` with
    respect to ``arg1`` and ``arg2`` in the above snippet. Note that, except the first argument, the
    keyword arguments of the ``optimality_fn`` should be a subset of the ones of ``solver_fn``.
    **In best practice, the ``optimality_fn`` should have the same signature as ``solver_fn``.**

    Args:
        optimality_fn: (callable)
            An equation function, ``optimality_fn(params, *args)``. The invariant is
            ``optimality_fn(solution, *args) == 0`` at the solution / root of ``solution``.
        argnums: (int or tuple of int, default: :const:`0`)
            Specifies arguments to compute gradients with respect to. The ``argnums`` can be an
            integer or a tuple of integers, which respect to the zero-based indices of the arguments
            of the ``solver_fn(params, *args)`` function. The argument ``params`` is included
            for the counting, while it is indexed as ``argnums=0``.
        has_aux: (default: :data:`False`)
            Whether the decorated solver function returns auxiliary data.
        solve: (callable, optional, default: :func:`linear_solve.solve_normal_cg`)
            a linear solver of the form ``solve(matvec, b)``.

    Returns:
        A solver function decorator, i.e., ``custom_root(optimality_fn)(solver_fn)``.
    """
    if isinstance(argnums, int):
        assert argnums != 0
        argnums = (argnums,)
    else:
        assert 0 not in argnums

    return functools.partial(
        _custom_root,
        optimality_fn=optimality_fn,
        solve=solve,
        argnums=argnums,
        has_aux=has_aux,
    )
