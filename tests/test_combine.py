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

import torchopt
from torchopt.alias.utils import _set_use_chain_flat


def test_chain() -> None:
    assert torchopt.chain() == torchopt.base.identity()
    assert torchopt.chain(torchopt.base.identity()) == torchopt.base.identity()
    assert (
        torchopt.chain(torchopt.base.identity(), torchopt.base.identity())
        == torchopt.base.identity()
    )
    assert torchopt.base.identity().chain(torchopt.base.identity()) == torchopt.base.identity()
    assert isinstance(torchopt.base.identity(), torchopt.base.IdentityGradientTransformation)
    assert isinstance(
        torchopt.base.identity().chain(torchopt.base.identity()),
        torchopt.base.ChainedGradientTransformation,
    )

    _set_use_chain_flat(False)
    adam = torchopt.adam()
    assert isinstance(adam, torchopt.base.ChainedGradientTransformation)
    assert isinstance(
        adam.chain(torchopt.base.identity()),
        torchopt.base.ChainedGradientTransformation,
    )
    assert adam.chain(torchopt.base.identity()) == adam
    assert torchopt.base.identity().chain(adam) == adam
    assert torchopt.chain(torchopt.base.identity(), adam, torchopt.base.identity()) == adam
    _set_use_chain_flat(True)

    assert isinstance(adam, torchopt.base.GradientTransformation)
    assert isinstance(
        adam.chain(torchopt.base.identity()),
        torchopt.base.ChainedGradientTransformation,
    )
    assert adam.chain(torchopt.base.identity()) == adam
    assert torchopt.base.identity().chain(adam) == adam
    assert torchopt.chain(torchopt.base.identity(), adam, torchopt.base.identity()) == adam
