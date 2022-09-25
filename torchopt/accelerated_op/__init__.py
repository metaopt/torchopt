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
"""The accelerated Ops."""

from typing import Iterable, Optional, Union

import torch

from torchopt.accelerated_op.adam_op import AdamOp


def is_available(
    devices: Optional[Union[int, str, torch.device, Iterable[Union[int, str, torch.device]]]] = None
) -> bool:
    """Check the availability of accelerated optimizer."""
    op = AdamOp()

    if devices is None:
        devices = [torch.device('cuda'), torch.device('cpu')]
    elif isinstance(devices, torch.device):
        devices = [devices]
    elif isinstance(devices, (int, str)):
        devices = [torch.device(devices)]

    try:
        for device in devices:
            device = torch.device(device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                return False
            updates = torch.tensor(1.0, device=device)
            op(updates, updates, updates, 1)
        return True
    except BaseException:  # pylint: disable=broad-except
        return False
