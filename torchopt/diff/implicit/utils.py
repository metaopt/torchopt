# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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

"""Implicit Meta-Gradient."""

# pylint: disable=invalid-name

from __future__ import annotations

from typing import Callable

from torchopt import linear_solve
from torchopt.diff.implicit.decorator import Args, _root_vjp
from torchopt.typing import TensorOrTensors, TupleOfOptionalTensors, TupleOfTensors


__all__ = ['root_vjp']


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches
def root_vjp(
    optimality_fn: Callable[..., TensorOrTensors],
    solution: TupleOfTensors,
    args: Args,
    grad_outputs: TupleOfTensors,
    output_is_tensor: bool,
    argnums: tuple[int, ...],
    solve: Callable[..., TensorOrTensors] | None = None,
) -> TupleOfOptionalTensors:
    """Return vector-Jacobian product of a root.

    The root is the `solution` of ``optimality_fn(solution, *args) == 0``.

    Args:
        optimality_fun (callable): An equation function, ``optimality_fn(params, *args)``. The
            invariant is ``optimality_fn(solution, *args) == 0`` at ``solution``.
        solution (tuple of Tensors): solution / root of `optimality_fun`.
        args (Args): tuple containing the arguments with respect to which we wish to
            differentiate ``solution`` against.
        grad_outputs (tuple of Tensors): The "vector" in the vector-Jacobian product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don't require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
        output_is_tensor (bool): Whether the output of ``optimality_fn`` is a single tensor.
        argnums (int or tuple of int): Specifies arguments to compute gradients with respect to. The
            ``argnums`` can be an integer or a tuple of integers.
        solve (callable, optional): A linear solver of the form ``solve(matvec, b)``.
            (default: :func:`linear_solve.solve_normal_cg`)

    Returns:
        tuple of the same length as ``len(args)`` containing the vector-Jacobian products w.r.t.
        each argument. Each ``vjps[i]`` has the same pytree structure as
        ``args[i]``.
    """
    if solve is None:
        solve = linear_solve.solve_normal_cg()

    return _root_vjp(
        optimality_fn=optimality_fn,
        solution=solution,
        args=args,
        grad_outputs=grad_outputs,
        output_is_tensor=output_is_tensor,
        argnums=argnums,
        solve=solve,
    )
