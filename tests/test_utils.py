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

import torch

import torchopt
from torchopt import pytree


def test_stop_gradient() -> None:
    x = torch.tensor(1.0, requires_grad=True)
    y = 2 * x
    assert y.grad_fn is not None
    torchopt.stop_gradient(y)
    assert y.grad_fn is None
    fc = torch.nn.Linear(1, 1, False)
    fc._parameters['weight'] = fc.weight * 2
    assert fc.weight.grad_fn is not None
    torchopt.stop_gradient(fc)
    assert fc.weight.grad_fn is None


def test_module_clone() -> None:
    x = torch.tensor(1.0, requires_grad=True)
    y = 2 * x
    assert y.grad_fn is not None
    z = torchopt.module_clone(y, by='reference')
    assert z is y
    z = torchopt.module_clone(x, by='copy')
    assert z is not x
    assert z.grad_fn.next_functions[0][0].variable is x

    z = torchopt.module_clone(y, by='deepcopy')
    assert z is not y
    assert z.grad_fn is None
    assert torch.equal(z, y)

    x = torch.tensor(1.0, requires_grad=True)
    y = torchopt.module_clone(x, by='reference', device='meta')
    assert y.grad_fn.next_functions[0][0].variable is x
    assert y.is_meta

    y = torchopt.module_clone(x, by='copy', device='meta')
    assert y is not x
    assert y.grad_fn.next_functions[0][0].next_functions[0][0].variable is x
    assert y.is_meta

    y = torchopt.module_clone(x, by='deepcopy', device='meta')
    assert y is not x
    assert y.grad_fn is None
    assert y.is_meta

    if torch.cuda.is_available():
        x = torch.tensor(1.0, requires_grad=True)
        y = torchopt.module_clone(x, by='reference', device='cuda')
        assert y.grad_fn.next_functions[0][0].variable is x
        assert y.is_cuda

        y = torchopt.module_clone(x, by='copy', device='cuda')
        assert y is not x
        assert y.grad_fn.next_functions[0][0].next_functions[0][0].variable is x
        assert y.is_cuda

        y = torchopt.module_clone(x, by='deepcopy', device='cuda')
        assert y is not x
        assert y.grad_fn is None
        assert torch.equal(y.to(x.device), x)
        assert y.is_cuda


def test_extract_state_dict():
    fc = torch.nn.Linear(1, 1)
    state_dict = torchopt.extract_state_dict(fc, by='reference', device=torch.device('meta'))
    for param_dict in state_dict.params:
        for k, v in param_dict.items():
            assert v.is_meta
            assert v.grad_fn.next_functions[0][0].variable is fc._parameters[k]

    state_dict = torchopt.extract_state_dict(fc, by='copy', device=torch.device('meta'))
    for param_dict in state_dict.params:
        for k, v in param_dict.items():
            assert v.is_meta
            assert v.grad_fn.next_functions[0][0].next_functions[0][0].variable is fc._parameters[k]

    state_dict = torchopt.extract_state_dict(fc, by='deepcopy', device=torch.device('meta'))
    for param_dict in state_dict.params:
        for v in param_dict.values():
            assert v.is_meta
            assert v.grad_fn is None

    state_dict = torchopt.extract_state_dict(fc, by='reference')
    for param_dict in state_dict.params:
        for k, v in param_dict.items():
            assert v is fc._parameters[k]

    state_dict = torchopt.extract_state_dict(fc, by='copy')
    for param_dict in state_dict.params:
        for k, v in param_dict.items():
            assert torch.equal(v, fc._parameters[k])
            assert v.grad_fn.next_functions[0][0].variable is fc._parameters[k]

    state_dict = torchopt.extract_state_dict(fc, by='deepcopy')
    for param_dict in state_dict.params:
        for k, v in param_dict.items():
            assert torch.equal(v, fc._parameters[k])
            assert v.grad_fn is None

    optim = torchopt.MetaAdam(fc, 1.0)
    loss = fc(torch.ones(1, 1)).sum()
    optim.step(loss)
    state_dict = torchopt.extract_state_dict(optim)
    same = pytree.tree_map(lambda x, y: x is y, state_dict, tuple(optim.state_groups))
    assert all(pytree.tree_flatten(same)[0])


def test_stop_gradient_for_state_dict() -> None:
    fc = torch.nn.Linear(1, 1)

    state_dict = torchopt.extract_state_dict(fc, by='copy')
    for param_dict in state_dict.params:
        for k, v in param_dict.items():
            assert v.grad_fn.next_functions[0][0].variable is fc._parameters[k]

    torchopt.stop_gradient(state_dict)
    for param_dict in state_dict.params:
        for k, v in param_dict.items():
            assert v.grad_fn is None
            assert torch.equal(v, fc._parameters[k])
