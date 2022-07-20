# Copyright 2022 MetaOPT Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import helpers
from torch import nn
from torchrl.envs import GymEnv, ParallelEnv, SerialEnv
from torchrl.modules import OneHotCategorical, ProbabilisticTDModule


class CategoricalMLPPolicy(nn.Module):
    """Policy network based on a multi-layer perceptron (MLP), with a
    `Categorical` distribution output. This policy network can be used on tasks
    with discrete action spaces (eg. `TabularMDPEnv`).
    """

    def __init__(
        self,
        input_size,
        output_size,
    ):
        super(CategoricalMLPPolicy, self).__init__()
        self.torso = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(32, output_size)
        self.value_head = nn.Linear(32, 1)

    def forward(self, inputs, params=None):
        embedding = self.torso(inputs)
        logits = self.policy_head(embedding)
        values = self.value_head(embedding)
        return logits, values


NUM_STATES = 10
NUM_ACTIONS = 5
MAX_STEPS = 10

NUM_ENV = 4

if __name__ == '__main__':
    lambda_env = lambda: GymEnv(
        'TabularMDP-v0', num_states=NUM_STATES, num_actions=NUM_ACTIONS, max_episode_steps=MAX_STEPS
    )
    env = ParallelEnv(NUM_ENV, lambda_env)
    # env = SerialEnv(NUM_ENV, lambda_env)
    td = env.reset()

    print(env.action_spec)
    print(env.observation_spec)

    print(td.get('observation'))

    dummy_env = lambda_env()
    tasks = dummy_env.sample_tasks(3)

    policy_module = CategoricalMLPPolicy(env.observation_spec.shape[-1], env.action_spec.shape[-1])
    policy = ProbabilisticTDModule(
        module=policy_module,
        spec=env.action_spec,
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        in_keys=['observation'],
        out_keys=['action', 'state_value'],
    )

    # gathering rollouts
    for task in tasks:
        env.reset_task(task)
        print('random policy: ', env.rollout(n_steps=10))
        print('structured policy: ', env.rollout(policy=policy, n_steps=10))

    env.close()
