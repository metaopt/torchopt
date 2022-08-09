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

import inspect
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union
from typing import Tuple

import functorch
import jax
import torch
from torch.autograd import Function

from TorchOpt._src import linear_solve


def root_vjp(optimality_fun: Callable,
             sol: Any,
             args: Tuple,
             cotangent: Any,
             argnums: Tuple,
             solve: Callable = linear_solve.solve_normal_cg) -> Any:
    """Vector-Jacobian product of a root.

    The invariant is ``optimality_fun(sol, *args) == 0``.

    Args:
      optimality_fun: the optimality function to use.
      sol: solution / root (pytree).
      args: tuple containing the arguments with respect to which we wish to
        differentiate ``sol`` against.
      cotangent: vector to left-multiply the Jacobian with
        (pytree, same structure as ``sol``).
      solve: a linear solver of the form ``x = solve(matvec, b)``,
        where ``matvec(x) = Ax`` and ``Ax=b``.
    Returns:
      tuple of the same length as ``len(args)`` containing the vjps w.r.t.
      each argument. Each ``vjps[i]`` has the same pytree structure as
      ``args[i]``.
    """
    def fun_sol(sol):
        # We close over the arguments.
        return optimality_fun(sol, *args)

    _, vjp_fun_sol = functorch.vjp(fun_sol, sol)

    # Compute the multiplication A^T u = (u^T A)^T.
    def matvec(u): return vjp_fun_sol(u)[0]

    # The solution of A^T u = v, where
    # A = jacobian(optimality_fun, argnums=0)
    # v = -cotangent.
    v = jax.tree_util.tree_map(torch.neg, cotangent)
    u = solve(matvec, v)

    class MaskArgs:
        def __init__(self, argnums, *args) -> None:
            self.argnums = argnums
            pre_filled = []
            post_filled = []
            for idx, arg in enumerate(args):
                if idx + 1 in argnums:  # plus 1 because we exclude the first argument
                    post_filled.append(arg)
                else:
                    pre_filled.append(arg)
            self.pre_filled = pre_filled
            self.post_filled = post_filled
            self.len_args = len(self.pre_filled) + len(self.post_filled)

        def __call__(self, *args) -> Any:
            args = list(args)
            true_args = []
            pre_filled_counter = 0
            for idx in range(self.len_args):
                if idx + 1 in self.argnums:  # plus 1 because we exclude the first argument
                    arg = args[idx]
                else:
                    arg = self.pre_filled[pre_filled_counter]
                    pre_filled_counter += 1
                true_args.append(arg)
            return optimality_fun(sol, *true_args)

    # def fun_args(*args):
    #     # We close over the solution.
    #     return optimality_fun(sol, *args)

    fun_args = MaskArgs(argnums, *args)

    _, vjp_fun_args = functorch.vjp(fun_args, *fun_args.post_filled)

    result = vjp_fun_args(u)
    result = list(result)
    true_result = [None]
    for idx in range(fun_args.len_args):
        if idx + 1 in argnums:  # plus 1 because we exclude the first argument
            true_result.append(result[idx])
        else:
            true_result.append(None)

    return tuple(true_result)


def _extract_kwargs(kwarg_keys, flat_args):
    n = len(flat_args) - len(kwarg_keys)
    args, kwarg_vals = flat_args[:n], flat_args[n:]
    kwargs = dict(zip(kwarg_keys, kwarg_vals))
    return args, kwargs


def _signature_bind(signature, *args, **kwargs):
    ba = signature.bind(*args, **kwargs)
    ba.apply_defaults()
    return ba.args, ba.kwargs


def _signature_bind_and_match(signature, *args, **kwargs):
    # We want to bind *args and **kwargs based on the provided
    # signature, but also to associate the resulting positional
    # arguments back. To achieve this, we lift arguments to a triple:
    #
    #   (was_kwarg, ref, value)
    #
    # where ref is an index position (int) if the original argument was
    # from *args and a dictionary key if the original argument was from
    # **kwargs. After binding to the inspected signature, we use the
    # tags to associate the resolved positional arguments back to their
    # arg and kwarg source.

    args = [(False, i, v) for i, v in enumerate(args)]
    kwargs = {k: (True, k, v) for (k, v) in kwargs.items()}
    ba = signature.bind(*args, **kwargs)

    mapping = [(was_kwarg, ref) for was_kwarg, ref, _ in ba.args]

    def map_back(out_args):
        src_args = [None] * len(args)
        src_kwargs = {}
        for (was_kwarg, ref), out_arg in zip(mapping, out_args):
            if was_kwarg:
                src_kwargs[ref] = out_arg
            else:
                src_args[ref] = out_arg
        return src_args, src_kwargs

    out_args = tuple(v for _, _, v in ba.args)
    out_kwargs = {k: v for k, (_, _, v) in ba.kwargs.items()}
    return out_args, out_kwargs, map_back


def _split_tensor_and_others(mixed_tuple):
    flat_tuple, tree = jax.tree_flatten(mixed_tuple)
    tensor_tuple = []
    non_tensor_tuple = []
    tensor_mask = []
    for item in flat_tuple:
        if isinstance(item, torch.Tensor):
            tensor_tuple.append(item)
            tensor_mask.append(True)
        else:
            non_tensor_tuple.append(item)
            tensor_mask.append(False)
    return tree, tuple(tensor_mask), tuple(tensor_tuple), tuple(non_tensor_tuple)


def _merge_tensor_and_others(tree, tensor_mask, tensor_tuple, non_tensor_tuple):
    tensor_counter = 0
    non_tensor_counter = 0
    result_tuple = []
    for is_tensor in tensor_mask:
        if is_tensor:
            result_tuple.append(tensor_tuple[tensor_counter])
            tensor_counter += 1
        else:
            result_tuple.append(non_tensor_tuple[non_tensor_counter])
            non_tensor_counter += 1
    result_tuple = tuple(result_tuple)
    return tree.unflatten(result_tuple)

def _custom_root(solver_fun, optimality_fun, solve, argnums, has_aux,
                 reference_signature=None):
    # When caling through `jax.custom_vjp`, jax attempts to resolve all
    # arguments passed by keyword to positions (this is in order to
    # match against a `nondiff_argnums` parameter that we do not use
    # here). It does so by resolving them according to the custom_jvp'ed
    # function's signature. It disallows functions defined with a
    # catch-all `**kwargs` expression, since their signature cannot
    # always resolve all keyword arguments to positions.
    #
    # We can loosen the constraint on the signature of `solver_fun` so
    # long as we resolve keywords to positions ourselves. We can do so
    # just in time, by flattening the `kwargs` dict (respecting its
    # iteration order) and supplying `custom_vjp` with a
    # positional-argument-only function. We then explicitly coordinate
    # flattening and unflattening around the `custom_vjp` boundary.
    #
    # Once we make it past the `custom_vjp` boundary, we do some more
    # work to align arguments with the reference signature (which is, by
    # default, the signature of `optimality_fun`).

    solver_fun_signature = inspect.signature(solver_fun)

    if reference_signature is None:
        reference_signature = inspect.signature(optimality_fun)

    elif not isinstance(reference_signature, inspect.Signature):
        # If is a CompositeLinearFunction, accesses subfun.
        # Otherwise, assumes a Callable.
        fun = getattr(reference_signature, "subfun", reference_signature)
        reference_signature = inspect.signature(fun)

    def make_custom_vjp_solver_fun(solver_fun, kwarg_keys, args_sign):
        class ImplicitMetaGradient(Function):
            @staticmethod
            def forward(ctx, *flat_args):
                args = []
                for idx, (start_point, is_tuple) in enumerate(args_sign):
                    if is_tuple:
                        args.append(tuple(flat_args[start_point:args_sign[idx+1][0]]))
                    else:
                        args.append(flat_args[start_point])
                args = tuple(args)

                args, kwargs = _extract_kwargs(kwarg_keys, args)
                res = solver_fun(*args, **kwargs)
                args_tree, args_tensor_mask, args_tensor, args_non_tensor = _split_tensor_and_others(args)
                ctx.args_tree = args_tree
                ctx.args_tensor_mask = args_tensor_mask
                ctx.args_non_tensor = args_non_tensor
                if has_aux:
                    aux = res[1]
                    res = res[0]
                    if isinstance(res, torch.Tensor):
                        ctx.save_for_backward(res, *args_tensor)
                    else:
                        ctx.save_for_backward(*res, *args_tensor)
                    return *res, aux
                else:
                    if isinstance(res, torch.Tensor):
                        ctx.save_for_backward(res, *args_tensor)
                    else:
                        ctx.save_for_backward(*res, *args_tensor)
                    return res

            @staticmethod
            def backward(ctx, *cotangent):
                if has_aux:
                    cotangent = cotangent[:-1]

                saved_tensors = ctx.saved_tensors
                res, args_tensor = saved_tensors[:len(cotangent)], saved_tensors[len(cotangent):]
                args_tree = ctx.args_tree
                args_tensor_mask = ctx.args_tensor_mask
                args_non_tensor = ctx.args_non_tensor
                args = _merge_tensor_and_others(args_tree, args_tensor_mask, args_tensor, args_non_tensor)

                args, kwargs = _extract_kwargs(kwarg_keys, args)

                sol = res
                ba_args, ba_kwargs, map_back = _signature_bind_and_match(
                    reference_signature, *args, **kwargs)
                if ba_kwargs:
                    raise TypeError(
                        "keyword arguments to solver_fun could not be resolved to "
                        "positional arguments based on the signature "
                        f"{reference_signature}. This can happen under custom_root if "
                        "optimality_fun takes catch-all **kwargs, or under "
                        "custom_fixed_point if fixed_point_fun takes catch-all **kwargs, "
                        "both of which are currently unsupported.")

                # Compute VJPs w.r.t. args.
                vjps = root_vjp(optimality_fun=optimality_fun, sol=sol,
                                args=ba_args[1:], cotangent=cotangent, argnums=argnums, solve=solve)
                # Prepend None as the vjp for init_params.

                arg_vjps, kws_vjps = map_back(vjps)
                ordered_vjps = tuple(arg_vjps) + tuple(kws_vjps[k] for k in kwargs.keys())
                true_vjps = []
                for idx, ((_, is_tuple), vjp) in enumerate(zip(args_sign, ordered_vjps)):
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
                if isinstance(arg, torch.Tensor):
                    args_sign.append((args_counter, False))  # start position, is_tuple
                    flatten_args.append(arg)
                    args_counter += 1
                elif isinstance(arg, tuple):
                    args_sign.append((args_counter, True))  # start position, is_tuple
                    for arg_item in arg:
                        flatten_args.append(arg_item)
                    args_counter += len(arg)
                else:
                    raise RuntimeError("must be tensor or tensor tuple")
            else:
                args_sign.append((args_counter, False))  # start position, is_tuple
                flatten_args.append(arg)
                args_counter += 1

        args_sign = tuple(args_sign)
        flatten_args = tuple(flatten_args)

        result = make_custom_vjp_solver_fun(solver_fun, keys, args_sign).apply(*flatten_args, *vals)
        if has_aux:
            return tuple(result[:-1]), result[-1]
        else:
            return result

    return wrapped_solver_fun


def custom_root(optimality_fun: Callable,
                argnums: Union[tuple, int] = 0,
                has_aux: bool = False,
                solve: Callable = None,
                reference_signature: Optional[Callable] = None):
    """Decorator for adding implicit differentiation to a root solver.

    Args:
      optimality_fun: an equation function, ``optimality_fun(params, *args)``.
        The invariant is ``optimality_fun(sol, *args) == 0`` at the
        solution / root ``sol``.
      has_aux: whether the decorated solver function returns auxiliary data.
      solve: a linear solver of the form ``solve(matvec, b)``.
      reference_signature: optional function signature
        (i.e. arguments and keyword arguments), with which the
        solver and optimality functions are expected to agree. Defaults
        to ``optimality_fun``. It is required that solver and optimality
        functions share the same input signature, but both might be
        defined in such a way that the signature correspondence is
        ambiguous (e.g. if both accept catch-all ``**kwargs``). To
        satisfy custom_root's requirement, any function with an
        unambiguous signature can be provided here.

    Returns:
      A solver function decorator, i.e.,
      ``custom_root(optimality_fun)(solver_fun)``.
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    if solve is None:
        solve = linear_solve.solve_normal_cg

    def wrapper(solver_fun):
        return _custom_root(solver_fun, optimality_fun, solve, argnums, has_aux,
                            reference_signature)

    return wrapper


# def custom_root(F, linear_solver=linear_solver.solve_cg):
#     def custom_root_impl(inner_fn):
#         inner_fn_signature = inspect.signature(inner_fn)

#         class ImplicitMetaGradient(Function):
#             @staticmethod
#             def forward(ctx, init_inner=None, l2reg=None, data=None):
#                 with torch.no_grad():
#                     inner_output = inner_fn(init_inner, l2reg, data)
#                     ctx.data = inner_output, l2reg, data
#                     return inner_output

#             @staticmethod
#             def backward(ctx, grad_output):
#                 inner_output, l2reg, data = ctx.data
#                 ctx.data = None

#                 F_for_l2reg = partial(F, inner_output, data=data)
#                 v = jvp(F_for_l2reg, l2reg, torch.ones_like(l2reg))[1]

#                 F_for_param = partial(F, l2reg=l2reg, data=data)

#                 def matvec(u):
#                     return jvp(F_for_param, inner_output, u)[1]
#                 J = -linear_solver(matvec=matvec,
#                                    b=v,
#                                    init=torch.ones_like(v),
#                                    maxiter=20)

#                 # partial_F = partial(F, data=data)
#                 # j1, j2 = jacobian(partial_F, (inner_output, l2reg))
#                 # J = -torch.inverse(j1) @ j2

#                 gradient = grad_output.t() @ J
#                 return None, gradient, None

#         def wrapper(*args, **kwargs):
#             args, kwargs = _signature_bind(inner_fn_signature, *args, **kwargs)
#             keys, vals = list(kwargs.keys()), list(kwargs.values())
#             return ImplicitMetaGradient.apply(*args, *vals)

#         return wrapper

#     return custom_root_impl
