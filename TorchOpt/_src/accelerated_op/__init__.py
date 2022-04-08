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

from TorchOpt._src.accelerated_op.adam_op import AdamOp


def accelerated_op_available(devices=None):
    import torch
    op = AdamOp()
    if devices is None:
        devices = [torch.device("cuda"), torch.device("cpu")]
    elif isinstance(devices, torch.device):
        devices = [devices]
    try:
        for device in devices:
            updates = torch.tensor(1., device=device)
            op(updates, updates, updates, 1)
        return True
    except:
        return False
