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

from TorchOpt._src import (accelerated_op_available, clip, combine, hook, schedule,
                   visual)
from TorchOpt._src.alias import adam, rmsprop, sgd
from TorchOpt._src.Optimizer import SGD, Adam, Optimizer, RMSProp
from TorchOpt._src.MetaOptimizer import MetaAdam, MetaOptimizer, MetaRMSProp, MetaSGD
from TorchOpt._src.update import apply_updates
from TorchOpt._src.utils import extract_state_dict, recover_state_dict, stop_gradient

__version__ = "0.4.1"

__all__ = (
    "accelerated_op_available",
    "clip",
    "combine",
    "hook",
    "schedule",
    "visual",
    "adam",
    "rmsprop",
    "sgd",
    "MetaAdam",
    "MetaOptimizer",
    "MetaRMSProp",
    "MetaSGD",
    "SGD",
    "Adam",
    "Optimizer",
    "RMSProp",
    "apply_updates",
    "extract_state_dict",
    "recover_state_dict",
    "stop_gradient",
)