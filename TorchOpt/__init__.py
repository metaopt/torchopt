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

from ._src import combine
from ._src import clip
from ._src import visual
from ._src import hook
from ._src import schedule
from ._src.MetaOptimizer import MetaOptimizer, MetaSGD, MetaAdam
from ._src.Optimizer import Optimizer, SGD, Adam
from ._src.update import apply_updates
from ._src.alias import sgd, adam
from ._src.utils import stop_gradient, extract_state_dict, recover_state_dict
from ._src import accelerated_op_available
__version__ = "0.4.0"
