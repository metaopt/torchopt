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
# This file is modified from:
# https://github.com/google/jaxopt/blob/main/jaxopt/_src/implicit_diff.py
# ==============================================================================
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implicit Meta-Gradient."""

# pylint: disable=invalid-name

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, Sequence, Tuple

import functorch
import torch
from torch.autograd import Function

from torchopt import linear_solve, pytree
from torchopt.typing import (
    ListOfOptionalTensors,
    ListOfTensors,
    TensorOrTensors,
    TupleOfOptionalTensors,
    TupleOfTensors,
)


__all__ = ['custom_root']


Args = Tuple[Any, ...]
KwArgs = Dict[str, Any]


class MaskedOptimalityFn:  # pylint: disable=missing-class-docstring,too-few-public-methods
    def __init__(
        self,
        optimality_fn: Callable[..., TensorOrTensors],
        solution: TensorOrTensors,
        output_is_tensor: bool,
        argnums: tuple[int, ...],
        *args: Any,
    ) -> None:
        self.optimality_fn = optimality_fn
        self.solution = solution
        self.output_is_tensor = output_is_tensor
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

    def __call__(self, *args: Any) -> TensorOrTensors:
        true_args = []
        pre_filled_counter = 0
        for idx in range(self.len_args):
            if idx + 1 in self.argnums:  # plus 1 because we exclude the first argument
                arg = args[idx]
            else:
                arg = self.pre_filled[pre_filled_counter]
                pre_filled_counter += 1
            true_args.append(arg)
        if self.output_is_tensor:
            return self.optimality_fn(self.solution[0], *true_args)
        return self.optimality_fn(self.solution, *true_args)


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches
def _root_vjp(
    optimality_fn: Callable[..., TensorOrTensors],
    solution: TupleOfTensors,
    args: Args,
    grad_outputs: TupleOfTensors,
    output_is_tensor: bool,
    argnums: tuple[int, ...],
    solve: Callable[..., TensorOrTensors],
) -> TupleOfOptionalTensors:
    if output_is_tensor:

        def optimality_cond(solution: TupleOfTensors) -> TensorOrTensors:
            return optimality_fn(solution[0], *args)

    else:

        def optimality_cond(solution: TupleOfTensors) -> TensorOrTensors:
            return optimality_fn(solution, *args)

    _, optimality_cond_vjp_fn, *_ = functorch.vjp(optimality_cond, solution)

    # Compute the multiplication A^T u = (u^T A)^T.
    if output_is_tensor:

        def matvec(u: TupleOfTensors) -> TupleOfTensors:
            return optimality_cond_vjp_fn(u[0])[0]

    else:

        def matvec(u: TupleOfTensors) -> TupleOfTensors:
            return optimality_cond_vjp_fn(u)[0]

    # The solution of A^T u = v, where
    # A = jacobian(optimality_fn, argnums=0)
    # v = -grad_outputs.
    v: TupleOfTensors = pytree.tree_map(torch.neg, grad_outputs)  # type: ignore[arg-type,assignment]
    u: TupleOfTensors = solve(matvec, v)  # type: ignore[assignment]

    masked_optimality_fn = MaskedOptimalityFn(
        optimality_fn,
        solution,
        output_is_tensor,
        argnums,
        *args,
    )

    _, optimality_vjp_fn, *_ = functorch.vjp(
        masked_optimality_fn,
        *masked_optimality_fn.post_filled,
    )

    output: TupleOfTensors
    output = optimality_vjp_fn(u[0]) if output_is_tensor else optimality_vjp_fn(u)

    # Prepend None as the vjp for init_params.
    true_output: ListOfOptionalTensors = [None]
    for idx in range(masked_optimality_fn.len_args):
        if idx + 1 in argnums:  # plus 1 because we exclude the first argument
            true_output.append(output[idx])
        else:
            true_output.append(None)

    return tuple(true_output)


def _extract_kwargs(kwarg_keys: Sequence[str], flat_args: tuple[Any, ...]) -> tuple[Args, KwArgs]:
    nargs = len(flat_args) - len(kwarg_keys)
    args, kwarg_vals = flat_args[:nargs], flat_args[nargs:]
    kwargs = dict(zip(kwarg_keys, kwarg_vals))
    return args, kwargs


def _signature_bind(signature: inspect.Signature, *args: Any, **kwargs: Any) -> tuple[Args, KwArgs]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.args, bound.kwargs


def _signature_bind_and_match(
    signature: inspect.Signature,
    *args: Any,
    **kwargs: Any,
) -> tuple[Args, KwArgs, Callable[[Args], tuple[Args, KwArgs]]]:
    # We want to bind *args and **kwargs based on the provided signature, but also to associate the
    # resulting positional arguments back. To achieve this, we lift arguments to a triple:
    #
    #   (was_kwarg, ref, value)
    #
    # where ref is an index position (int) if the original argument was from *args and a dictionary
    # key if the original argument was from **kwargs. After binding to the inspected signature, we
    # use the tags to associate the resolved positional arguments back to their args and kwargs
    # source.

    args = [(False, i, v) for i, v in enumerate(args)]
    kwargs = {k: (True, k, v) for (k, v) in kwargs.items()}
    bound = signature.bind(*args, **kwargs)

    mapping = [(was_kwarg, ref) for was_kwarg, ref, _ in bound.args]

    def map_args_back(out_args: Args) -> tuple[Args, KwArgs]:
        src_args = [None] * len(args)
        src_kwargs = {}
        for (was_kwarg, ref), out_arg in zip(mapping, out_args):
            if was_kwarg:
                src_kwargs[ref] = out_arg
            else:
                src_args[ref] = out_arg
        return tuple(src_args), src_kwargs

    out_args = tuple(v for _, _, v in bound.args)
    out_kwargs = {k: v for k, (_, _, v) in bound.kwargs.items()}
    return out_args, out_kwargs, map_args_back


def _split_tensor_and_others(
    mixed_tuple: tuple[Any, ...],
) -> tuple[pytree.PyTreeSpec, tuple[bool, ...], TupleOfTensors, tuple[Any, ...]]:
    flattened: list[Any]
    flattened, treespec = pytree.tree_flatten(mixed_tuple, none_is_leaf=True)  # type: ignore[arg-type]
    tensors: ListOfTensors = []
    non_tensors: list[Any] = []
    is_tensor_mask: list[bool] = []
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
    is_tensor_mask: tuple[bool, ...],
    tensors: TupleOfTensors,
    non_tensors: tuple[Any, ...],
) -> tuple[Any, ...]:
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
    return pytree.tree_unflatten(treespec, results)  # type: ignore[return-value]


# pylint: disable-next=too-many-arguments,too-many-statements
def _custom_root(
    solver_fn: Callable[..., TensorOrTensors | tuple[TensorOrTensors, Any]],
    optimality_fn: Callable[..., TensorOrTensors],
    solve: Callable[..., TensorOrTensors],
    argnums: tuple[int, ...],
    has_aux: bool,
    reference_signature: inspect.Signature | Callable | None = None,
) -> Callable[..., TensorOrTensors | tuple[TensorOrTensors, Any]]:
    solver_fn_signature = inspect.signature(solver_fn)

    if reference_signature is None:
        reference_signature = inspect.signature(optimality_fn)
    elif not isinstance(reference_signature, inspect.Signature):
        # If is a CompositeLinearFunction, accesses subfn.
        # Otherwise, assumes a Callable.
        fn = getattr(reference_signature, 'subfn', reference_signature)
        reference_signature = inspect.signature(fn)

    def make_custom_vjp_solver_fn(
        solver_fn: Callable[..., TensorOrTensors | tuple[TensorOrTensors, Any]],
        kwarg_keys: Sequence[str],
        args_signs: tuple[tuple[int, int, type[tuple | list] | None], ...],
    ) -> type[Function]:
        # pylint: disable-next=missing-class-docstring,abstract-method
        class ImplicitMetaGradient(Function):
            @staticmethod
            def forward(  # type: ignore[override] # pylint: disable=arguments-differ
                ctx: Any,
                *flat_args: Any,
            ) -> tuple[Any, ...]:
                output, aux, output_is_tensor = None, None, False

                args = []
                for offset, nargs, arg_seq_type in args_signs:
                    if arg_seq_type is not None:
                        args.append(arg_seq_type(flat_args[offset : offset + nargs]))
                    else:
                        args.append(flat_args[offset])
                args = tuple(args)

                args, kwargs = _extract_kwargs(kwarg_keys, args)
                output = solver_fn(*args, **kwargs)
                if has_aux:
                    if not (isinstance(output, tuple) and len(output) == 2):
                        raise RuntimeError(
                            f'custom_root(optimality_fn)(solver_fn)(*args): output of function '
                            f'solver_fn should be a tuple: (output, aux) if has_aux is True. '
                            f'Got {output}',
                        )
                    output, aux = output
                if isinstance(output, torch.Tensor):
                    output_is_tensor = True
                    output = (output,)
                elif not (isinstance(output, tuple) and all(map(torch.is_tensor, output))):
                    raise RuntimeError(
                        f'custom_root(optimality_fn)(solver_fn)(*args): output of function '
                        f'solver_fn should be a torch.Tensor or a tuple of torch.Tensor. '
                        f'Got {output}',
                    )
                output = tuple(t.data for t in output)

                (
                    args_treespec,
                    args_is_tensor_mask,
                    args_tensors,
                    args_non_tensors,
                ) = _split_tensor_and_others(args)
                ctx.args_treespec = args_treespec
                ctx.args_is_tensor_mask = args_is_tensor_mask
                ctx.args_non_tensors = args_non_tensors

                ctx.save_for_backward(*output, *args_tensors)
                ctx.output_is_tensor = output_is_tensor

                return (*output, aux, output_is_tensor, type(output))

            @staticmethod
            def backward(  # pylint: disable=too-many-locals
                ctx: Any,
                *grad_outputs: Any,
            ) -> TupleOfTensors:
                grad_outputs: TupleOfTensors = grad_outputs[:-3]

                saved_tensors = ctx.saved_tensors
                output = saved_tensors[: len(grad_outputs)]
                args_tensors = saved_tensors[len(grad_outputs) :]
                args_treespec = ctx.args_treespec
                args_is_tensor_mask = ctx.args_is_tensor_mask
                args_non_tensors = ctx.args_non_tensors
                args = _merge_tensor_and_others(
                    args_treespec,
                    args_is_tensor_mask,
                    args_tensors,
                    args_non_tensors,
                )

                args, kwargs = _extract_kwargs(kwarg_keys, args)

                bound_args, bound_kwargs, map_args_back = _signature_bind_and_match(
                    reference_signature,  # type: ignore[arg-type]
                    *args,
                    **kwargs,
                )
                if bound_kwargs:
                    raise TypeError(
                        f'keyword arguments to solver_fn could not be resolved to positional '
                        f'arguments based on the signature {reference_signature}. This can '
                        f'happen under custom_root if optimality_fn takes catch-all **kwargs, or '
                        f'under custom_fixed_point if fixed_point_fn takes catch-all **kwargs, '
                        f'both of which are currently unsupported.',
                    )

                # Compute VJPs w.r.t. args.
                vjps = _root_vjp(
                    optimality_fn=optimality_fn,
                    solution=output,
                    args=bound_args[1:],
                    grad_outputs=grad_outputs,
                    output_is_tensor=ctx.output_is_tensor,
                    argnums=argnums,
                    solve=solve,
                )

                args_vjps, kwargs_vjps = map_args_back(vjps)
                ordered_vjps = tuple(args_vjps) + tuple(kwargs_vjps[k] for k in kwargs)
                true_vjps = []
                for (_, _, arg_seq_type), vjp in zip(args_signs, ordered_vjps):
                    if arg_seq_type is not None:
                        true_vjps.extend(vjp)
                    else:
                        true_vjps.append(vjp)
                return tuple(true_vjps)

        return ImplicitMetaGradient

    @functools.wraps(solver_fn)
    def wrapped_solver_fn(
        *args: Any,
        **kwargs: Any,
    ) -> TensorOrTensors | tuple[TensorOrTensors, Any]:
        args, kwargs = _signature_bind(solver_fn_signature, *args, **kwargs)
        keys, vals = list(kwargs.keys()), list(kwargs.values())

        args_signs: list[tuple[int, int, type[tuple | list] | None]] = []
        flat_args: list[Any] = []
        args_offset = 0
        for idx, arg in enumerate(args):
            if idx in argnums:
                if isinstance(arg, torch.Tensor):
                    args_signs.append((args_offset, 1, None))  # start position, None
                    flat_args.append(arg)
                    args_offset += 1
                elif isinstance(arg, (tuple, list)) and all(map(torch.is_tensor, arg)):
                    nargs = len(arg)
                    args_signs.append(
                        (args_offset, nargs, type(arg)),  # start position, sequence type
                    )
                    flat_args.extend(arg)
                    args_offset += nargs
                else:
                    raise RuntimeError(
                        'custom_root(optimality_fn)(solver_fn)(*args): argument of function '
                        'solver_fn specified with `argnums` should be a torch.Tensor or a tuple of '
                        'torch.Tensor',
                    )
            else:
                args_signs.append((args_offset, 1, None))  # start position, None
                flat_args.append(arg)
                args_offset += 1

        args_signs = tuple(args_signs)
        flat_args = tuple(flat_args)

        result = make_custom_vjp_solver_fn(solver_fn, keys, args_signs).apply(*flat_args, *vals)
        *output, aux, output_is_tensor, output_type = result
        output = output[0] if output_is_tensor else output_type(output)
        if has_aux:
            return output, aux
        return output

    return wrapped_solver_fn


def custom_root(
    optimality_fn: Callable[..., TensorOrTensors],
    argnums: int | tuple[int, ...],
    has_aux: bool = False,
    solve: Callable[..., TensorOrTensors] | None = None,
) -> Callable[
    [Callable[..., TensorOrTensors | tuple[TensorOrTensors, Any]]],
    Callable[..., TensorOrTensors | tuple[TensorOrTensors, Any]],
]:
    """Return a decorator for adding implicit differentiation to a root solver.

    This wrapper should be used as a decorator:

    .. code-block:: python

        def optimality_fn(optimal_params, ...):
            ...
            return residual

        @custom_root(optimality_fn, argnums=argnums)
        def solver_fn(params, arg1, arg2, ...):
            ...
            return optimal_params

        optimal_params = solver_fn(init_params, ...)

    The first argument to ``optimality_fn`` and ``solver_fn`` is preserved as the parameter input.
    The ``argnums`` argument refers to the indices of the variables in ``solver_fn``'s signature.
    For example, setting ``argnums=(1, 2)`` will compute the gradient of ``optimal_params`` with
    respect to ``arg1`` and ``arg2`` in the above snippet. Note that, except the first argument, the
    keyword arguments of the ``optimality_fn`` should be a subset of the ones of ``solver_fn``.
    **In best practice, the ``optimality_fn`` should have the same signature as ``solver_fn``.**

    Args:
        optimality_fn (callable): An equation function, ``optimality_fn(params, *args)``. The
            invariant is ``optimality_fn(solution, *args) == 0`` at the solution / root of
            ``solution``.
        argnums (int or tuple of int): Specifies arguments to compute gradients with respect to. The
            ``argnums`` can be an integer or a tuple of integers, which respect to the zero-based
            indices of the arguments of the ``solver_fn(params, *args)`` function. The argument
            ``params`` is included for the counting, while it is indexed as ``argnums=0``.
        has_aux (bool, optional): Whether the decorated solver function returns auxiliary data.
            (default: :data:`False`)
        solve (callable, optional): A linear solver of the form ``solve(matvec, b)``.
            (default: :func:`linear_solve.solve_normal_cg`)

    Returns:
        A solver function decorator, i.e., ``custom_root(optimality_fn)(solver_fn)``.
    """
    if isinstance(argnums, int):
        assert argnums != 0
        argnums = (argnums,)
    else:
        assert 0 not in argnums

    if solve is None:
        solve = linear_solve.solve_normal_cg()

    return functools.partial(
        _custom_root,
        optimality_fn=optimality_fn,
        solve=solve,
        argnums=argnums,
        has_aux=has_aux,
    )
