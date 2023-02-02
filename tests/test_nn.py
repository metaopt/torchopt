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


def test_delattr() -> None:
    m = torchopt.nn.MetaGradientModule()
    x = torch.nn.Linear(1, 1)
    m.add_module('x', x)
    delattr(m, 'x')
    assert not hasattr(m, 'x')
    m.add_meta_module('x', x)
    delattr(m, 'x')
    assert not hasattr(m, 'x')
    x = torch.tensor(1.0, requires_grad=True)
    m.register_parameter('x', x)
    delattr(m, 'x')
    assert not hasattr(m, 'x')
    x = torch.tensor(1.0, requires_grad=True)
    m.register_meta_parameter('x', x)
    delattr(m, 'x')
    assert not hasattr(m, 'x')
    m.register_buffer('x', x)
    assert len(m._buffers) == 1
    assert m.x is x
    delattr(m, 'x')
    assert len(m._buffers) == 0
    assert not hasattr(m, 'x')


def test_setattr() -> None:
    m = torchopt.nn.MetaGradientModule()
    x_set = torch.nn.Linear(1, 1)
    m.x = x_set
    assert m.x is x_set
    x_set = torch.tensor(1.0, requires_grad=True)
    m.x = x_set
    assert m.x is x_set

    bn = torch.nn.BatchNorm2d(128)
    m.bn = bn
    assert m.bn is bn


def test_register_meta_parameter() -> None:
    x = torch.tensor(1.0, requires_grad=True)
    m = torchopt.nn.MetaGradientModule()
    m.register_meta_parameter('x', x)
    assert m.x is x


def test_add_meta_module() -> None:
    m = torchopt.nn.MetaGradientModule()
    m1 = torch.nn.Linear(1, 1)
    m.add_module('m', m1)
    assert m.m is m1
    delattr(m, 'm')
    m.register_meta_module('m', m1)
    assert m.m is m1


def test_meta_module() -> None:
    m = torchopt.nn.MetaGradientModule()
    meta_module = torch.nn.Linear(1, 1)
    m.add_meta_module('m', meta_module)
    assert next(m.named_meta_modules())[1] is meta_module
    assert next(m.named_meta_children())[1] is meta_module
    assert next(m.meta_children()) is meta_module
    assert next(m.meta_modules()) is meta_module


def test_add_meta_parameters() -> None:
    m = torchopt.nn.MetaGradientModule()
    x = torch.tensor(1.0, requires_grad=True)
    m.register_meta_parameter('x', x)
    assert next(m.named_meta_parameters())[1] is x


def test_named_modules() -> None:
    m = torchopt.nn.MetaGradientModule()
    assert next(m.named_modules())[1] is m
