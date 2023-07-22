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
"""Differentiable Meta-Optimizers."""

from torchopt.optim.meta.adadelta import MetaAdaDelta, MetaAdadelta
from torchopt.optim.meta.adagrad import MetaAdaGrad, MetaAdagrad
from torchopt.optim.meta.adam import MetaAdam
from torchopt.optim.meta.adamax import MetaAdaMax, MetaAdamax
from torchopt.optim.meta.adamw import MetaAdamW
from torchopt.optim.meta.base import MetaOptimizer
from torchopt.optim.meta.radam import MetaRAdam
from torchopt.optim.meta.rmsprop import MetaRMSProp, MetaRMSprop
from torchopt.optim.meta.sgd import MetaSGD
