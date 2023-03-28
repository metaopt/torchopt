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
# This file is modified from:
# https://github.com/alexis-jacq/LOLA_DiCE
# ==============================================================================

import gym
import numpy as np
from gym.spaces import Discrete, Tuple


class OneHot(gym.Space):
    """
    One-hot space. Used as the observation space.
    """

    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.multinomial(1, [1.0 / self.n] * self.n)

    def contains(self, x):
        return (
            isinstance(x, np.ndarray)
            and x.shape == (self.n,)
            and np.all(np.logical_or(x == 0, x == 1))
            and np.sum(x) == 1
        )

    @property
    def shape(self):
        return (self.n,)

    def __repr__(self):
        return 'OneHot(%d)' % self.n

    def __eq__(self, other):
        return self.n == other.n


class IPD(gym.Env):
    """
    A two-agent vectorized environment.
    Possible actions for each agent are Cooperate (C) and Defect (D).
    """

    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat = np.array([[-2, 0], [-3, -1]])
        self.states = np.array([[1, 2], [3, 4]])

        self.action_space = Tuple([Discrete(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)])
        self.observation_space = Tuple([OneHot(self.NUM_STATES) for _ in range(self.NUM_AGENTS)])
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int) for _ in range(self.NUM_AGENTS)
        ]

        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.batch_size)
        observation = [init_state, init_state]
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observation, info

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1

        r0 = self.payout_mat[ac0, ac1]
        r1 = self.payout_mat[ac1, ac0]
        s0 = self.states[ac0, ac1]
        s1 = self.states[ac1, ac0]
        observation = [s0, s1]
        reward = [r0, r1]
        done = self.step_count == self.max_steps
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observation, reward, done, info
