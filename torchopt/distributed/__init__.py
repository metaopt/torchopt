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
"""Distributed utilities."""

import torch.distributed as dist
import torch.distributed.rpc as rpc

from torchopt.distributed import api, autograd, world
from torchopt.distributed.api import *  # noqa: F403
from torchopt.distributed.world import *  # noqa: F403


__all__ = ['is_available', *api.__all__, *world.__all__]


def is_available() -> bool:
    """Check if the distributed module is available."""
    return dist.is_available() and rpc.is_available() and autograd.is_available()
