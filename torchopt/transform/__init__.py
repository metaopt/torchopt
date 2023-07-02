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
# This file is modified from:
# https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
# ==============================================================================
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Preset transformations."""

from torchopt.transform.add_decayed_weights import add_decayed_weights, masked
from torchopt.transform.nan_to_num import nan_to_num
from torchopt.transform.scale import scale
from torchopt.transform.scale_by_adadelta import scale_by_adadelta
from torchopt.transform.scale_by_adam import scale_by_accelerated_adam, scale_by_adam
from torchopt.transform.scale_by_adamax import scale_by_adamax
from torchopt.transform.scale_by_radam import scale_by_radam
from torchopt.transform.scale_by_rms import scale_by_rms
from torchopt.transform.scale_by_rss import scale_by_rss
from torchopt.transform.scale_by_schedule import scale_by_schedule
from torchopt.transform.scale_by_stddev import scale_by_stddev
from torchopt.transform.trace import trace


__all__ = [
    'trace',
    'scale',
    'scale_by_schedule',
    'add_decayed_weights',
    'masked',
    'scale_by_adam',
    'scale_by_adamax',
    'scale_by_adadelta',
    'scale_by_radam',
    'scale_by_accelerated_adam',
    'scale_by_rss',
    'scale_by_rms',
    'scale_by_stddev',
    'nan_to_num',
]
