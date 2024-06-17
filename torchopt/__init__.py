# Copyright 2022-2024 MetaOPT Team. All Rights Reserved.
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

from torchopt import (
    accelerated_op,
    alias,
    base,
    clip,
    combine,
    diff,
    distributed,
    hook,
    linalg,
    linear_solve,
    nn,
    optim,
    pytree,
    schedule,
    typing,
    visual,
)
from torchopt.accelerated_op import is_available as accelerated_op_available
from torchopt.alias import adadelta, adagrad, adam, adamax, adamw, radam, rmsprop, sgd
from torchopt.clip import clip_grad_norm
from torchopt.combine import chain
from torchopt.hook import register_hook
from torchopt.optim import (
    SGD,
    AdaDelta,
    Adadelta,
    AdaGrad,
    Adagrad,
    Adam,
    AdaMax,
    Adamax,
    AdamW,
    Optimizer,
    RAdam,
    RMSProp,
    RMSprop,
)
from torchopt.optim.func import FuncOptimizer
from torchopt.optim.meta import (
    MetaAdaDelta,
    MetaAdadelta,
    MetaAdaGrad,
    MetaAdagrad,
    MetaAdam,
    MetaAdaMax,
    MetaAdamax,
    MetaAdamW,
    MetaOptimizer,
    MetaRAdam,
    MetaRMSProp,
    MetaRMSprop,
    MetaSGD,
)
from torchopt.transform import nan_to_num
from torchopt.update import apply_updates
from torchopt.utils import (
    extract_state_dict,
    module_clone,
    module_detach_,
    recover_state_dict,
    stop_gradient,
)
from torchopt.version import __version__


__all__ = [
    'SGD',
    'AdaDelta',
    'AdaGrad',
    'AdaMax',
    'Adadelta',
    'Adagrad',
    'Adam',
    'AdamW',
    'Adamax',
    'FuncOptimizer',
    'MetaAdaDelta',
    'MetaAdaGrad',
    'MetaAdaMax',
    'MetaAdadelta',
    'MetaAdagrad',
    'MetaAdam',
    'MetaAdamW',
    'MetaAdamax',
    'MetaOptimizer',
    'MetaRAdam',
    'MetaRMSProp',
    'MetaRMSprop',
    'MetaSGD',
    'Optimizer',
    'RAdam',
    'RMSProp',
    'RMSprop',
    'accelerated_op_available',
    'adadelta',
    'adagrad',
    'adam',
    'adamax',
    'adamw',
    'apply_updates',
    'chain',
    'clip_grad_norm',
    'extract_state_dict',
    'module_clone',
    'module_detach_',
    'nan_to_num',
    'radam',
    'recover_state_dict',
    'register_hook',
    'rmsprop',
    'sgd',
    'stop_gradient',
]
