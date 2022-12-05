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
"""Base class for neural network modules that hold meta-parameters and meta-modules."""

from collections import OrderedDict
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from torchopt import pytree


class MetaInputsContainer(NamedTuple):
    """Container for parameters and modules in the constructor input arguments."""

    meta_parameters: Set[torch.Tensor]
    meta_modules: Set[nn.Module]


class MetaGradientModule(nn.Module):  # pylint: disable=abstract-method
    """Base class for neural network modules that hold meta-parameters and meta-modules."""

    _meta_inputs: MetaInputsContainer
    _meta_parameters: Dict[str, Optional[torch.Tensor]]
    _meta_modules: Dict[str, Optional[nn.Module]]

    def __new__(cls, *args, **kwargs) -> 'MetaGradientModule':
        """Creates a new module instance."""
        instance = super().__new__(cls)
        flat_args: List[Any]
        flat_args = pytree.tree_leaves((args, kwargs))  # type: ignore[arg-type]
        meta_parameters = {x for x in flat_args if isinstance(x, torch.Tensor) and x.requires_grad}
        meta_modules = {x for x in flat_args if isinstance(x, nn.Module) and x.training}
        for meta_module in tuple(meta_modules):
            meta_parameters.update(meta_module.parameters())
            meta_modules.update(meta_module.modules())

        instance._meta_inputs = MetaInputsContainer(meta_parameters, meta_modules)
        instance._meta_parameters: Dict[str, Optional[torch.Tensor]] = OrderedDict()  # type: ignore[misc]
        instance._meta_modules: Dict[str, Optional[nn.Module]] = OrderedDict()  # type: ignore[misc]
        return instance

    def __getattr__(self, name: str) -> Union[torch.Tensor, nn.Module]:
        """Gets an attribute of the module."""
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_meta_parameters' in self.__dict__:
            _meta_parameters = self.__dict__['_meta_parameters']
            if name in _meta_parameters:
                return _meta_parameters[name]
        if '_meta_modules' in self.__dict__:
            _meta_modules = self.__dict__['_meta_modules']
            if name in _meta_modules:
                return _meta_modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # pylint: disable-next=too-many-branches,too-many-statements
    def __setattr__(self, name: str, value: Union[torch.Tensor, nn.Module]) -> None:
        """Sets an attribute of the module."""

        def remove_from(*dicts_or_sets):
            for dict_or_set in dicts_or_sets:
                if name in dict_or_set:
                    if isinstance(dict_or_set, dict):
                        del dict_or_set[name]
                    else:
                        dict_or_set.discard(name)

        params = self.__dict__.get('_parameters')
        meta_params = self.__dict__.get('_meta_parameters')
        if isinstance(value, torch.Tensor) and value.requires_grad:
            if params is None:
                raise AttributeError('cannot assign parameters before Module.__init__() call')
            if meta_params is None:
                raise AttributeError(
                    'cannot assign meta-parameters before MetaGradientModule.__init__() call'
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
                self._meta_parameters,
                self._meta_modules,
            )
            if value in self._meta_inputs.meta_parameters:
                self.register_meta_parameter(name, value)
            else:
                self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{torch.typename(value)}' as parameter '{name}' "
                    f'(torch.Tensor or None expected)'
                )
            self.register_parameter(name, value)  # type: ignore[unreachable]
        elif meta_params is not None and name in meta_params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{torch.typename(value)}' as meta-parameter '{name}' "
                    f'(torch.Tensor or None expected)'
                )
            self.register_meta_parameter(name, value)  # type: ignore[unreachable]
        else:
            modules = self.__dict__.get('_modules')
            meta_modules = self.__dict__.get('_meta_modules')
            if isinstance(value, nn.Module):
                if modules is None:
                    raise AttributeError('cannot assign module before Module.__init__() call')
                if meta_modules is None:
                    raise AttributeError(
                        'cannot assign module before MetaGradientModule.__init__() call'
                    )
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                    self._meta_parameters,
                    self._meta_modules,
                )
                if value in self._meta_inputs.meta_modules:
                    meta_modules[name] = value
                else:
                    modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"cannot assign '{torch.typename(value)}' as child module '{name}' "
                        f'(torch.nn.Module or None expected)'
                    )
                modules[name] = value  # type: ignore[unreachable]
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"cannot assign '{torch.typename(value)}' as buffer '{name}' "
                            f'(torch.Tensor or None expected)'
                        )
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Deletes an attribute of the module."""
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        elif name in self._meta_parameters:
            del self._meta_parameters[name]
        elif name in self._meta_modules:
            del self._meta_modules[name]
        else:
            object.__delattr__(self, name)

    def register_parameter(self, name: str, param: Optional[torch.Tensor]) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (torch.Tensor or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError('cannot assign parameter before Module.__init__() call')
        if not isinstance(name, str):
            raise TypeError(f'parameter name should be a string. Got {torch.typename(name)}')
        if '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        if name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        if hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._parameters[name] = None
            return

        if not isinstance(param, torch.Tensor):
            raise TypeError(
                f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
                f'(torch.Tensor or None required)'
            )
        if not param.requires_grad:
            raise ValueError(
                f"cannot assign Tensor that `requires_grad=False` to parameter '{name}'"
            )
        if param in self._meta_inputs.meta_parameters:
            raise ValueError(
                f"cannot assign Tensor that is a meta-parameter to parameter '{name}'. "
                f'Use self.register_meta_parameter() instead.'
            )

        self._parameters[name] = param  # type: ignore

    def register_meta_parameter(self, name: str, param: Optional[torch.Tensor]) -> None:
        r"""Adds a meta-parameter to the module.

        The meta-parameter can be accessed as an attribute using given name.

        Args:
            name (string): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (torch.Tensor or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if '_meta_parameters' not in self.__dict__:
            raise AttributeError(
                'cannot assign meta-parameter before MetaGradientModule.__init__() call'
            )
        if not isinstance(name, str):
            raise TypeError(f'meta-parameter name should be a string. Got {torch.typename(name)}')
        if '.' in name:
            raise KeyError("meta-parameter name can't contain \".\"")
        if name == '':
            raise KeyError("meta-parameter name can't be empty string \"\"")
        if hasattr(self, name) and name not in self._meta_parameters:
            raise KeyError(f"attribute '{name}' already exists")

        if param is None:
            self._meta_parameters[name] = None
            return

        if not isinstance(param, torch.Tensor):
            raise TypeError(
                f"cannot assign '{torch.typename(param)}' object to meta-parameter '{name}' "
                f'(torch.Tensor or None required)'
            )
        if not param.requires_grad:
            raise ValueError(
                f"cannot assign Tensor that `requires_grad=False` to meta-parameter '{name}'"
            )

        self._meta_parameters[name] = param

    def add_module(self, name: str, module: Optional[nn.Module]) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, nn.Module) and module is not None:
            raise TypeError(f'{torch.typename(module)} is not a Module subclass')
        if not isinstance(name, str):
            raise TypeError(f'module name should be a string. Got {torch.typename(name)}')
        if hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        if '.' in name:
            raise KeyError(f"module name can't contain \".\", got: {name}")
        if name == '':
            raise KeyError("module name can't be empty string \"\"")
        if module in self._meta_inputs.meta_modules:
            raise ValueError(
                f"cannot add module that is a meta-module to module '{name}'. "
                f'Use self.add_meta_module() instead.'
            )

        self._modules[name] = module

    def register_module(self, name: str, module: Optional[nn.Module]) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def add_meta_module(self, name: str, meta_module: Optional[nn.Module]) -> None:
        r"""Adds a child meta-module to the current module.

        The meta-module can be accessed as an attribute using the given name.

        Args:
            name (string): name of the child meta-module. The child meta-module can be
                accessed from this module using the given name
            meta_module (Module): child meta-module to be added to the module.
        """
        if not isinstance(meta_module, nn.Module) and meta_module is not None:
            raise TypeError(f'{torch.typename(meta_module)} is not a Module subclass')
        if not isinstance(name, str):
            raise TypeError(f'meta-module name should be a string. Got {torch.typename(name)}')
        if hasattr(self, name) and name not in self._meta_modules:
            raise KeyError(f"attribute '{name}' already exists")
        if '.' in name:
            raise KeyError(f"meta-module name can't contain \".\", got: {name}")
        if name == '':
            raise KeyError("meta-module name can't be empty string \"\"")

        self._meta_modules[name] = meta_module

    def register_meta_module(self, name: str, meta_module: Optional[nn.Module]) -> None:
        r"""Alias for :func:`add_meta_module`."""
        self.add_meta_module(name, meta_module)

    def meta_parameters(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        r"""Returns an iterator over module meta-parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module and
                all submodules. Otherwise, yields only meta-parameters that
                are direct members of this module.

        Yields:
            Parameter: module meta-parameter

        Example::

            >>> for param in model.meta_parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, meta_param in self.named_meta_parameters(recurse=recurse):
            yield meta_param

    def named_meta_parameters(
        self, prefix: str = '', recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        r"""Returns an iterator over module meta-parameters, yielding both the name of the meta-parameter as well as the meta-parameter itself.

        Args:
            prefix (str): prefix to prepend to all meta-parameter names.
            recurse (bool): if True, then yields meta-parameters of this module
                and all submodules. Otherwise, yields only meta-parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, meta_param in self.named_meta_parameters():
            >>>    if name in ['bias']:
            >>>        print(meta_param.size())

        """  # pylint: disable=line-too-long
        memo = set()
        for name, param in getattr(self, '_meta_parameters', {}).items():
            if param is None or param in memo:
                continue
            memo.add(param)
            yield prefix + name, param
        for name, meta_module in getattr(self, '_meta_modules', {}).items():
            if meta_module is None:
                continue
            submodule_prefix = prefix + name
            yield from meta_module.named_parameters(submodule_prefix, recurse)

    def meta_children(self) -> Iterator[nn.Module]:
        r"""Returns an iterator over immediate children meta-modules.

        Yields:
            Module: a child meta-module
        """
        for _, module in self.named_meta_children():
            yield module

    def named_meta_children(self) -> Iterator[Tuple[str, nn.Module]]:
        r"""Returns an iterator over immediate children meta-modules, yielding both the name of the meta-module as well as the meta-module itself.

        Yields:
            (string, Module): Tuple containing a name and child meta-module

        Example::

            >>> for name, meta_module in model.named_meta_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(meta_module)

        """  # pylint: disable=line-too-long
        memo = set()
        for name, meta_module in self._meta_modules.items():
            if meta_module is not None and meta_module not in memo:
                memo.add(meta_module)
                yield name, meta_module

    def meta_modules(self) -> Iterator[nn.Module]:
        r"""Returns an iterator over all meta-modules in the network.

        Yields:
            Module: a meta-module in the network

        Note:
            Duplicate meta-modules are returned only once.
        """
        for _, meta_module in self.named_meta_modules():
            yield meta_module

    def named_meta_modules(
        self, memo: Optional[Set[nn.Module]] = None, prefix: str = '', remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Module]]:
        r"""Returns an iterator over all meta-modules in the network, yielding both the name of the meta-module as well as the meta-module itself.

        Args:
            memo: a memo to store the set of meta-modules already added to the result
            prefix: a prefix that will be added to the name of the meta-module
            remove_duplicate: whether to remove the duplicated meta-module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and meta-module

        Note:
            Duplicate modules are returned only once.
        """  # pylint: disable=line-too-long
        if memo is None:
            memo = set()
        if self in memo:
            return

        if remove_duplicate:
            memo.add(self)

        for name, meta_module in self._meta_modules.items():
            if meta_module is None:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            yield from meta_module.named_modules(memo, submodule_prefix, remove_duplicate)
