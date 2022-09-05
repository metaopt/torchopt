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
"""TorchOpt: a high-performance optimizer library built upon PyTorch."""

from torchopt._src import accelerated_op_available, clip, combine, hook, schedule, visual
from torchopt._src.alias import adam, adamw, rmsprop, sgd
from torchopt._src.clip import clip_grad_norm
from torchopt._src.combine import chain
from torchopt._src.optimizer import SGD, Adam, AdamW, Optimizer, RMSProp, RMSprop, meta
from torchopt._src.optimizer.meta import (
    MetaAdam,
    MetaAdamW,
    MetaOptimizer,
    MetaRMSProp,
    MetaRMSprop,
    MetaSGD,
)
from torchopt._src.update import apply_updates
from torchopt._src.utils import extract_state_dict, recover_state_dict, stop_gradient
from torchopt.version import __version__


__all__ = [
    'accelerated_op_available',
    'clip',
    'combine',
    'hook',
    'schedule',
    'visual',
    'adam',
    'adamw',
    'rmsprop',
    'sgd',
    'clip_grad_norm',
    'chain',
    'Optimizer',
    'SGD',
    'Adam',
    'AdamW',
    'RMSProp',
    'RMSprop',
    'MetaOptimizer',
    'MetaSGD',
    'MetaAdam',
    'MetaAdamW',
    'MetaRMSProp',
    'MetaRMSprop',
    'apply_updates',
    'extract_state_dict',
    'recover_state_dict',
    'stop_gradient',
]
