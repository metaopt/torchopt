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
"""Implicit Meta-Gradient."""

from torchopt.diff.implicit import nn
from torchopt.diff.implicit.decorator import custom_root, root_vjp
from torchopt.diff.implicit.nn import ImplicitMetaGradientModule


__all__ = ['custom_root', 'ImplicitMetaGradientModule', 'root_vjp']
