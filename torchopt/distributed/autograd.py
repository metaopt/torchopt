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
"""Distributed Autograd."""

from threading import Lock
from typing import Optional, overload

import torch
import torch.distributed.autograd as autograd
from torch.distributed.autograd import context

from torchopt.typing import TensorOrTensors, TupleOfOptionalTensors, TupleOfTensors


__all__ = ['is_available', 'context']


LOCK = Lock()


def is_available():
    """Check if distributed autograd module is available."""
    return autograd.is_available()


if is_available():
    # pylint: disable-next=unused-import,ungrouped-imports
    from torch.distributed.autograd import DistAutogradContext, get_gradients

    def backward(
        autograd_ctx_id: int,
        tensors: TensorOrTensors,
        retain_graph: bool = False,
        inputs: Optional[TensorOrTensors] = None,
    ) -> None:
        """Perform distributed backward pass for local parameters.

        Computes the sum of gradients of given tensors with respect to graph leaves.

        Args:
            autograd_ctx_id: The autograd context id.
            tensors (Sequence[Tensor] or Tensor): Tensors of which the derivative will be computed.
            retain_graph (bool, optional): If :data:`False`, the graph used to compute the grad will
                be freed. Note that in nearly all cases setting this option to :data:`True` is not
                needed and often can be worked around in a much more efficient way.
            inputs (Sequence[Tensor] or Tensor, optional): Inputs w.r.t. which the gradient be will
                accumulated into ``.grad``. All other Tensors will be ignored. If not provided, the
                gradient is accumulated into all the leaf Tensors that were used to compute the
                attr::tensors.
        """
        if inputs is not None:
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            elif len(inputs) == 0:
                raise RuntimeError("'inputs' argument to backward() cannot be empty.")
            else:
                inputs = tuple(inputs)
            if not all(map(lambda t: t.requires_grad, inputs)):
                raise RuntimeError('One of the differentiated Tensors does not require grad')

        roots = [tensors] if isinstance(tensors, torch.Tensor) else list(tensors)
        autograd.backward(autograd_ctx_id, roots=roots, retain_graph=retain_graph)

        all_local_grads = autograd.get_gradients(autograd_ctx_id)
        if inputs is not None:
            inputs = set(inputs)  # type: ignore[assignment]
            all_local_grads = {p: g for p, g in all_local_grads.items() if p in inputs}

        with LOCK:
            for p, g in all_local_grads.items():
                if p.grad is not None:
                    p.grad = p.grad.add(g)
                else:
                    p.grad = g

    @overload
    def grad(
        autograd_ctx_id: int,
        outputs: TensorOrTensors,
        inputs: TensorOrTensors,
        retain_graph: bool = False,
    ) -> TupleOfTensors:
        ...

    @overload
    def grad(
        autograd_ctx_id: int,
        outputs: TensorOrTensors,
        inputs: TensorOrTensors,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> TupleOfOptionalTensors:
        ...

    def grad(
        autograd_ctx_id: int,
        outputs: TensorOrTensors,
        inputs: TensorOrTensors,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> TupleOfOptionalTensors:
        """Computes and returns the sum of gradients of outputs with respect to the inputs.

        Args:
            autograd_ctx_id: The autograd context id.
            outputs (sequence of Tensor): outputs of the differentiated function.
            inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be returned (and not
                accumulated into ``.grad``).
            retain_graph (bool, optional): If :data:`False`, the graph used to compute the grad will
                be freed. Note that in nearly all cases setting this option to :data:`True` is not
                needed and often can be worked around in a much more efficient way.
            allow_unused (bool, optional): If :data:`False`, specifying inputs that were not used
                when computing outputs (and therefore their grad is always zero) is an error.
                Defaults to :data:`False`.
        """
        outputs = [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)
        inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
        if not all(map(lambda t: t.requires_grad, inputs)):
            raise RuntimeError('One of the differentiated Tensors does not require grad')

        autograd.backward(autograd_ctx_id, roots=outputs, retain_graph=retain_graph)

        all_local_grads = autograd.get_gradients(autograd_ctx_id)
        grads = []
        for p in inputs:
            try:
                grads.append(all_local_grads[p])
            except KeyError as ex:
                if not allow_unused:
                    raise RuntimeError(
                        'One of the differentiated Tensors appears to not have been used in the '
                        'graph. Set allow_unused=True if this is the desired behavior.'
                    ) from ex
                grads.append(None)  # type: ignore[arg-type]

        return tuple(grads)

    __all__.extend(['DistAutogradContext', 'get_gradients', 'backward', 'grad'])
