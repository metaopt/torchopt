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
from typing import Tuple

import jax
import torch
from TorchOpt._src import linear_solver
from torch.autograd import Function
from torch.autograd.functional import jvp, vjp
import functorch
from functools import partial


def root_vjp(optimality_fun: Callable,
             sol: Any,
             args: Tuple,
             cotangent: Any,
             mask: Tuple,
             solve: Callable = linear_solver.solve_normal_cg) -> Any:
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
    v = jax.tree_map(torch.neg, cotangent)
    u = solve(matvec, v)

    class MaskArgs:
        def __init__(self, mask, *args) -> None:
            self.mask = mask
            pre_filled = []
            post_filled = []
            for is_tensor, arg in zip(mask, args):
                if is_tensor:
                    post_filled.append(arg)
                else:
                    pre_filled.append(arg)
            self.pre_filled = pre_filled
            self.post_filled = post_filled

        def __call__(self, *args) -> Any:
            args = list(args)
            true_args = []
            for is_tensor in self.mask:
                if is_tensor:
                    arg = args.pop(0)
                else:
                    arg = self.pre_filled.pop(0)
                true_args.append(arg)
            return optimality_fun(sol, *true_args)

    # def fun_args(*args):
    #     # We close over the solution.
    #     return optimality_fun(sol, *args)

    fun_args = MaskArgs(mask[1:], *args)

    _, vjp_fun_args = functorch.vjp(fun_args, *fun_args.post_filled)

    result = vjp_fun_args(u)
    result = list(result)
    true_result = [None]
    for is_tensor in mask[1:]:
        if not is_tensor:
            true_result.append(None)
        else:
            true_result.append(result.pop(0))
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


def _custom_root(solver_fun, optimality_fun, solve, has_aux,
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

    def make_custom_vjp_solver_fun(solver_fun, kwarg_keys):
        class ImplicitMetaGradient(Function):
            @staticmethod
            def forward(ctx, *flat_args):
                args, kwargs = _extract_kwargs(kwarg_keys, flat_args)
                ctx.mask = tuple([isinstance(term, torch.Tensor) for term in flat_args])
                res = solver_fun(*args, **kwargs)
                ctx.aux = (res, flat_args)
                return res

            @staticmethod
            def backward(ctx, cotangent):
                res, flat_args = ctx.aux
                ctx.aux = None
                mask = ctx.mask
                ctx.mask = None
                args, kwargs = _extract_kwargs(kwarg_keys, flat_args)

                # solver_fun can return auxiliary data if has_aux = True.
                if has_aux:
                    cotangent = cotangent[0]
                    sol = res[0]
                else:
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
                                args=ba_args[1:], cotangent=cotangent, mask=mask, solve=solve)
                # Prepend None as the vjp for init_params.

                arg_vjps, kws_vjps = map_back(vjps)
                ordered_vjps = tuple(arg_vjps) + tuple(kws_vjps[k] for k in kwargs.keys())
                return ordered_vjps
        return ImplicitMetaGradient

    def wrapped_solver_fun(*args, **kwargs):
        args, kwargs = _signature_bind(solver_fun_signature, *args, **kwargs)
        keys, vals = list(kwargs.keys()), list(kwargs.values())
        return make_custom_vjp_solver_fun(solver_fun, keys).apply(*args, *vals)

    return wrapped_solver_fun


def custom_root(optimality_fun: Callable,
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
    if solve is None:
        solve = linear_solver.solve_normal_cg

    def wrapper(solver_fun):
        return _custom_root(solver_fun, optimality_fun, solve, has_aux,
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
