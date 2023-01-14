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
"""Utility functions for stateless module calls."""

import contextlib
from typing import Dict, Generator, Iterable, Tuple, Union

import torch
import torch.nn as nn


__all__ = ['swap_state', 'reparameterize']


MISSING: torch.Tensor = object()  # type: ignore[assignment]


def swap_state(
    module: nn.Module,
    named_tensors: Union[Dict[str, torch.Tensor], Iterable[Tuple[str, torch.Tensor]]],
    allow_missing: bool = False,
) -> Dict[str, torch.Tensor]:
    """Swap the module parameters and/or buffers."""
    if not isinstance(named_tensors, dict):
        named_tensors = dict(named_tensors)

    def recursive_setattr(mod: nn.Module, path: str, value: torch.Tensor) -> torch.Tensor:
        """Set attribute recursively."""
        attr, dot, suffix = path.partition('.')
        if dot:
            return recursive_setattr(getattr(mod, attr), suffix, value)

        if allow_missing:
            orig = getattr(mod, attr, MISSING)
        else:
            orig = getattr(mod, attr)

        # pylint: disable=protected-access
        if value is MISSING:
            delattr(mod, attr)
        elif hasattr(mod, '_parameters') and attr in mod._parameters:
            mod._parameters[attr] = value  # type: ignore[assignment]
        elif hasattr(mod, '_buffers') and attr in mod._buffers:
            mod._buffers[attr] = value
        elif hasattr(mod, '_meta_parameters') and attr in mod._meta_parameters:  # type: ignore[operator]
            mod._meta_parameters[attr] = value  # type: ignore[operator,index]
        else:
            setattr(mod, attr, value)
        # pylint: enable=protected-access

        return orig

    orig_named_tensors = {
        name: recursive_setattr(module, name, tensor) for name, tensor in named_tensors.items()
    }
    return orig_named_tensors


@contextlib.contextmanager
def reparameterize(
    module: nn.Module,
    named_tensors: Union[Dict[str, torch.Tensor], Iterable[Tuple[str, torch.Tensor]]],
    allow_missing: bool = False,
) -> Generator[nn.Module, None, None]:
    """Reparameterize the module parameters and/or buffers."""
    if not isinstance(named_tensors, dict):
        named_tensors = dict(named_tensors)

    orig_named_tensors = {}
    try:
        orig_named_tensors = swap_state(module, named_tensors, allow_missing=allow_missing)
        yield module
    finally:
        swap_state(module, orig_named_tensors, allow_missing=allow_missing)
