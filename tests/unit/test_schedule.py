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

import unittest

import torchopt


class TestSchedule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.init_value = 1.
        cls.end_value = 0.
        cls.gap_value = cls.init_value - cls.end_value
        cls.transition_steps = 10
        cls.transition_begin = 1

    def setUp(self) -> None:
        pass

    def test_linear(self) -> None:
        schedule = torchopt.schedule.linear_schedule(
            init_value=self.init_value,
            end_value=self.end_value,
            transition_steps=self.transition_steps,
            transition_begin=self.transition_begin
        )
        for i in range(self.transition_begin, self.transition_steps):
            lr = schedule(i)
            lr_gt = self.init_value - self.gap_value * \
                (i - self.transition_begin) / self.transition_steps
            self.assertEqual(lr, lr_gt)


if __name__ == '__main__':
    unittest.main()
