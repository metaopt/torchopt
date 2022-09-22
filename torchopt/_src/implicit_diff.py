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

# pylint: disable=invalid-name

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import functorch
import torch
from torch.autograd import Function

from torchopt._src import linear_solve
from torchopt._src.utils import pytree


ARGS = Tuple[Any, ...]
KWARGS = Dict[Any, Any]


# pylint: disable-next=too-many-arguments,too-many-locals
def _root_vjp(
    optimality_fun: Callable,
    sol: Any,
    args: ARGS,
    cotangent: Any,
    res_is_tensor: bool,
    argnums: Tuple[int, ...],
    solve: Callable = linear_solve.solve_normal_cg(),
) -> Tuple[Any, ...]:
    def fun_sol(sol):
        # We close over the arguments.
        if res_is_tensor:
            return optimality_fun(sol[0], *args)
        return optimality_fun(sol, *args)

    _, vjp_fun_sol, *_ = functorch.vjp(fun_sol, sol)

    # Compute the multiplication A^T u = (u^T A)^T.
    def matvec(u):
        if res_is_tensor:
            return vjp_fun_sol(u[0])[0]
        return vjp_fun_sol(u)[0]

    # The solution of A^T u = v, where
    # A = jacobian(optimality_fun, argnums=0)
    # v = -cotangent.
    v = pytree.tree_map(torch.neg, cotangent)
    u = solve(matvec, v)

    class MaskArgs:  # pylint: disable=missing-class-docstring,too-few-public-methods
        def __init__(self, argnums: Tuple[int, ...], *args) -> None:
            self.argnums = argnums

            self.pre_filled = []
            self.post_filled = []
            for idx, arg in enumerate(args):
                if idx + 1 in argnums:  # plus 1 because we exclude the first argument
                    self.post_filled.append(arg)
                else:
                    self.pre_filled.append(arg)
            self.len_args = len(self.pre_filled) + len(self.post_filled)

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
            if res_is_tensor:
                return optimality_fun(sol[0], *true_args)
            return optimality_fun(sol, *true_args)

    fun_args = MaskArgs(argnums, *args)

    _, vjp_fun_args, *_ = functorch.vjp(fun_args, *fun_args.post_filled)

    if res_is_tensor:
        result = vjp_fun_args(u[0])
    else:
        result = vjp_fun_args(u)

    true_result = [None]
    for idx in range(fun_args.len_args):
        if idx + 1 in argnums:  # plus 1 because we exclude the first argument
            true_result.append(result[idx])
        else:
            true_result.append(None)

    return tuple(true_result)


def _extract_kwargs(kwarg_keys: Sequence[Any], flat_args: Tuple[Any, ...]) -> Tuple[ARGS, KWARGS]:
    nargs = len(flat_args) - len(kwarg_keys)
    args, kwarg_vals = flat_args[:nargs], flat_args[nargs:]
    kwargs = dict(zip(kwarg_keys, kwarg_vals))
    return args, kwargs


def _signature_bind(signature: inspect.Signature, *args, **kwargs) -> Tuple[ARGS, KWARGS]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.args, bound.kwargs


def _signature_bind_and_match(
    signature: inspect.Signature, *args, **kwargs
) -> Tuple[ARGS, KWARGS, Callable[[ARGS], Tuple[ARGS, KWARGS]]]:
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
    mixed_tuple: Tuple,
) -> Tuple[pytree.PyTreeDef, Tuple[bool, ...], Tuple[torch.Tensor, ...], Tuple[Any, ...]]:
    flattened, treedef = pytree.tree_flatten(mixed_tuple)
    tensors = []
    non_tensors = []
    is_tensor_mask = []
    for item in flattened:
        if torch.is_tensor(item):
            tensors.append(item.data)
            is_tensor_mask.append(True)
        else:
            non_tensors.append(item)
            is_tensor_mask.append(False)
    return treedef, tuple(is_tensor_mask), tuple(tensors), tuple(non_tensors)


def _merge_tensor_and_others(
    treedef: pytree.PyTreeDef,
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
    return pytree.tree_unflatten(treedef, results)


# pylint: disable-next=too-many-arguments,too-many-statements
def _custom_root(
    solver_fun: Callable,
    optimality_fun: Callable,
    solve: Callable,
    argnums: Tuple[int, ...],
    has_aux: bool,
    reference_signature: Optional[Union[inspect.Signature, Callable]] = None,
) -> Callable:
    solver_fun_signature = inspect.signature(solver_fun)

    if reference_signature is None:
        reference_signature = inspect.signature(optimality_fun)
    elif not isinstance(reference_signature, inspect.Signature):
        # If is a CompositeLinearFunction, accesses subfun.
        # Otherwise, assumes a Callable.
        fun = getattr(reference_signature, 'subfun', reference_signature)
        reference_signature = inspect.signature(fun)

    def make_custom_vjp_solver_fun(solver_fun, kwarg_keys, args_sign):
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
                res = solver_fun(*args, **kwargs)
                (
                    args_treedef,
                    args_is_tensor_mask,
                    args_tensors,
                    args_non_tensors,
                ) = _split_tensor_and_others(args)
                ctx.args_treedef = args_treedef
                ctx.args_is_tensor_mask = args_is_tensor_mask
                ctx.args_non_tensors = args_non_tensors
                if has_aux:
                    aux = res[1]
                    res = res[0]
                    if torch.is_tensor(res):
                        ctx.save_for_backward(res, *args_tensors)
                    else:
                        ctx.save_for_backward(*res, *args_tensors)
                    ctx.res_is_tensor = torch.is_tensor(res)
                    return res + (aux,)

                if torch.is_tensor(res):
                    ctx.save_for_backward(res, *args_tensors)
                else:
                    ctx.save_for_backward(*res, *args_tensors)
                ctx.res_is_tensor = torch.is_tensor(res)
                return res

            @staticmethod
            def backward(ctx, *cotangent):  # pylint: disable=too-many-locals
                if has_aux:
                    cotangent = cotangent[:-1]

                saved_tensors = ctx.saved_tensors
                res, args_tensors = saved_tensors[: len(cotangent)], saved_tensors[len(cotangent) :]
                args_treedef = ctx.args_treedef
                args_is_tensor_mask = ctx.args_is_tensor_mask
                args_non_tensors = ctx.args_non_tensors
                args = _merge_tensor_and_others(
                    args_treedef, args_is_tensor_mask, args_tensors, args_non_tensors
                )

                args, kwargs = _extract_kwargs(kwarg_keys, args)

                sol = res
                bound_args, bound_kwargs, map_args_back = _signature_bind_and_match(
                    reference_signature, *args, **kwargs  # type: ignore[arg-type]
                )
                if bound_kwargs:
                    raise TypeError(
                        'keyword arguments to solver_fun could not be resolved to '
                        'positional arguments based on the signature '
                        f'{reference_signature}. This can happen under custom_root if '
                        'optimality_fun takes catch-all **kwargs, or under '
                        'custom_fixed_point if fixed_point_fun takes catch-all **kwargs, '
                        'both of which are currently unsupported.'
                    )

                # Compute VJPs w.r.t. args.
                vjps = _root_vjp(
                    optimality_fun=optimality_fun,
                    sol=sol,
                    args=bound_args[1:],
                    cotangent=cotangent,
                    res_is_tensor=ctx.res_is_tensor,
                    argnums=argnums,
                    solve=solve,
                )
                # Prepend None as the vjp for init_params.

                arg_vjps, kwargs_vjps = map_args_back(vjps)
                ordered_vjps = tuple(arg_vjps) + tuple(kwargs_vjps[k] for k in kwargs.keys())
                true_vjps = []
                for (_, is_tuple), vjp in zip(args_sign, ordered_vjps):
                    if is_tuple:
                        for item in vjp:
                            true_vjps.append(item)
                    else:
                        true_vjps.append(vjp)
                return tuple(true_vjps)

        return ImplicitMetaGradient

    def wrapped_solver_fun(*args, **kwargs):
        args, kwargs = _signature_bind(solver_fun_signature, *args, **kwargs)
        keys, vals = list(kwargs.keys()), list(kwargs.values())

        args_sign = []
        flatten_args = []
        args_counter = 0
        for idx, arg in enumerate(args):
            if idx in argnums:
                if torch.is_tensor(arg):
                    args_sign.append((args_counter, False))  # start position, is_tuple
                    flatten_args.append(arg)
                    args_counter += 1
                elif isinstance(arg, tuple):
                    args_sign.append((args_counter, True))  # start position, is_tuple
                    for arg_item in arg:
                        flatten_args.append(arg_item)
                    args_counter += len(arg)
                else:
                    raise RuntimeError('must be tensor or tensor tuple')
            else:
                args_sign.append((args_counter, False))  # start position, is_tuple
                flatten_args.append(arg)
                args_counter += 1

        args_sign = tuple(args_sign)
        flatten_args = tuple(flatten_args)

        result = make_custom_vjp_solver_fun(solver_fun, keys, args_sign).apply(*flatten_args, *vals)
        if has_aux:
            return result[:-1], result[-1]
        return result

    return wrapped_solver_fun


def custom_root(
    optimality_fun: Callable,
    argnums: Union[int, Tuple[int, ...]] = 0,
    has_aux: bool = False,
    solve: Callable = linear_solve.solve_normal_cg(),
    reference_signature: Optional[Union[inspect.Signature, Callable]] = None,
) -> Callable[[Callable], Callable]:
    """Decorator for adding implicit differentiation to a root solver.

    Args:
        optimality_fun: (callable)
            An equation function, ``optimality_fun(params, *args)``. The invariant is
            ``optimality_fun(sol, *args) == 0`` at the solution / root of ``sol``.
        argnums: (int or tuple of int, default: :const:`0`)
            Specifies arguments to compute gradients with respect to.
        has_aux: (default: :data:`False`)
            Whether the decorated solver function returns auxiliary data.
        solve: (callable, optional, default: :func:`linear_solve.solve_normal_cg`)
            a linear solver of the form ``solve(matvec, b)``.
        reference_signature: (function signature, optional)
            Function signature (i.e. arguments and keyword arguments), with which the solver and
            optimality functions are expected to agree. Defaults to ``optimality_fun``. It is
            required that solver and optimality functions share the same input signature, but both
            might be defined in such a way that the signature correspondence is ambiguous (e.g. if
            both accept catch-all ``**kwargs``). To satisfy ``custom_root``'s requirement, any
            function with an unambiguous signature can be provided here.

    Returns:
        A solver function decorator, i.e., ``custom_root(optimality_fun)(solver_fun)``.
    """
    if isinstance(argnums, int):
        assert argnums != 0
        argnums = (argnums,)
    else:
        assert 0 not in argnums

    return functools.partial(
        _custom_root,
        optimality_fun=optimality_fun,
        solve=solve,
        argnums=argnums,
        has_aux=has_aux,
        reference_signature=reference_signature,
    )
