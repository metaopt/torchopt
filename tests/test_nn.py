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

import re

import pytest
import torch
import torch.nn as nn

import helpers
import torchopt


def test_property() -> None:
    m = torchopt.nn.MetaGradientModule()
    x = helpers.get_model()
    m.add_module('x', x)
    assert m.x is x
    delattr(m, 'x')
    assert not hasattr(m, 'x')
    m.add_meta_module('x', x)
    assert m.x is x
    delattr(m, 'x')
    assert not hasattr(m, 'x')
    x = torch.tensor(1.0, requires_grad=True)
    m.register_parameter('x', x)
    assert m.x is x
    delattr(m, 'x')
    assert not hasattr(m, 'x')
    x = torch.tensor(1.0, requires_grad=True)
    m.register_meta_parameter('x', x)
    assert m.x is x
    delattr(m, 'x')
    assert not hasattr(m, 'x')
    m.register_buffer('x', x)
    assert len(m._buffers) == 1
    assert m.x is x
    delattr(m, 'x')
    assert len(m._buffers) == 0
    assert not hasattr(m, 'x')


def test_register_tensors() -> None:
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(1.0, requires_grad=True)
    z = torch.tensor(1.0, requires_grad=False)
    b = torch.tensor(1.0, requires_grad=False)

    m = torchopt.nn.MetaGradientModule()
    m.register_meta_parameter('x', x)
    assert m.x is x

    m = torchopt.nn.MetaGradientModule(x)
    m.x = x
    m.y = y
    m.z = z

    assert m._meta_parameters['x'] is x
    assert m._parameters['y'] is y
    assert (
        hasattr(m, 'z')
        and m.z is z
        and 'z' not in m._meta_parameters
        and 'z' not in m._parameters
        and 'z' not in m._buffers
    )

    del m.x
    object.__setattr__(m, 'x', x)
    assert hasattr(m, 'x') and m.x is x and 'x' not in m._meta_parameters
    m.x = x
    assert m._meta_parameters['x'] is x

    m.register_buffer('b', None)
    assert m.b is None
    m.b = b
    assert m.b is b and 'b' in m._buffers

    m = torchopt.nn.MetaGradientModule(x, b)

    with pytest.raises(
        TypeError,
        match=re.escape('parameter name should be a string. Got bytes'),
    ):
        m.register_meta_parameter(b'x', x)

    with pytest.raises(
        KeyError,
        match=re.escape("parameter name can't contain '.'"),
    ):
        m.register_meta_parameter('x.x', x)

    with pytest.raises(
        KeyError,
        match=re.escape("parameter name can't be empty string ''"),
    ):
        m.register_meta_parameter('', x)

    m.register_buffer('z', None)
    with pytest.raises(
        KeyError,
        match=re.escape("attribute 'z' already exists"),
    ):
        m.register_meta_parameter('z', x)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "cannot assign Tensor that is a meta-parameter to parameter 'x'. "
            'Use self.register_meta_parameter() instead.',
        ),
    ):
        m.register_parameter('x', x)

    m.x = x
    with pytest.raises(
        KeyError,
        match=re.escape("attribute 'x' already exists"),
    ):
        m.register_parameter('x', x)

    with pytest.raises(
        TypeError,
        match=re.escape('parameter name should be a string. Got bytes'),
    ):
        m.register_parameter(b'y', y)

    with pytest.raises(
        KeyError,
        match=re.escape("parameter name can't contain '.'"),
    ):
        m.register_parameter('y.x', y)

    with pytest.raises(
        KeyError,
        match=re.escape("parameter name can't be empty string ''"),
    ):
        m.register_parameter('', y)


def test_no_super_init() -> None:
    class NoSuper1(torchopt.nn.MetaGradientModule):
        def __init__(self, x) -> None:
            self.x = x

    with pytest.raises(
        AttributeError,
        match=re.escape('cannot assign parameters before Module.__init__() call'),
    ):
        NoSuper1(torch.tensor(1.0, requires_grad=True))

    class NoSuper2(torchopt.nn.MetaGradientModule):
        def __init__(self) -> None:
            self.x = torch.tensor(1.0, requires_grad=True)

    with pytest.raises(
        AttributeError,
        match=re.escape('cannot assign parameters before Module.__init__() call'),
    ):
        NoSuper2()

    class NoSuper3(torchopt.nn.MetaGradientModule):
        def __init__(self) -> None:
            self.register_buffer('x', torch.tensor(1.0))

    with pytest.raises(
        AttributeError,
        match=re.escape('cannot assign buffer before Module.__init__() call'),
    ):
        NoSuper3()

    class NoSuper4(torchopt.nn.MetaGradientModule):
        def __init__(self) -> None:
            self.x = torch.tensor(1.0, requires_grad=False)

    NoSuper4()  # no error

    class NoSuper5(torchopt.nn.MetaGradientModule):
        def __init__(self, x) -> None:
            self.x = x

    with pytest.raises(
        AttributeError,
        match=re.escape('cannot assign module before Module.__init__() call'),
    ):
        NoSuper5(nn.Linear(1, 1))

    class NoSuper6(torchopt.nn.MetaGradientModule):
        def __init__(self) -> None:
            self.x = nn.Linear(1, 1)

    with pytest.raises(
        AttributeError,
        match=re.escape('cannot assign module before Module.__init__() call'),
    ):
        NoSuper6()


def test_add_meta_module() -> None:
    meta_module = helpers.get_model()
    fc = nn.Linear(1, 1)

    m = torchopt.nn.MetaGradientModule(meta_module)
    m.fc = fc
    assert m.fc is fc
    assert m._modules['fc'] is fc

    m.meta = meta_module
    assert m.meta is meta_module
    assert m._meta_modules['meta'] is meta_module

    assert all(p1 is p2 for p1, p2 in zip(m.parameters(), fc.parameters()))
    assert all(p1 is p2 for p1, p2 in zip(m.meta_parameters(), meta_module.parameters()))

    m = torchopt.nn.MetaGradientModule(meta_module)
    m.add_meta_module('fc', fc)
    assert m.fc is fc
    assert all(p1 is p2 for p1, p2 in zip(m.meta_parameters(), fc.parameters()))


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
