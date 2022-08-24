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

import numpy as np

import torchopt


def test_linear_schedule() -> None:
    init_value = 1.0
    end_value = 0.0
    gap_value = init_value - end_value
    transition_steps = 10
    transition_begin = 1

    schedule = torchopt.schedule.linear_schedule(
        init_value=init_value,
        end_value=end_value,
        transition_steps=transition_steps,
        transition_begin=transition_begin,
    )
    for i in range(transition_begin, transition_steps):
        lr = schedule(i)
        lr_gt = init_value - gap_value * (i - transition_begin) / transition_steps
        assert np.allclose(lr, lr_gt)
