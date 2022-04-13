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

import argparse
from typing import NamedTuple

import gym
import torch
import torch.optim as optim
import numpy as np

import TorchOpt
from helpers.policy import CategoricalMLPPolicy
import functorch

TASK_NUM = 40
TRAJ_NUM = 20
TRAJ_LEN = 10

STATE_DIM = 10
ACTION_DIM = 5

GAMMA = 0.99
LAMBDA = 0.95

outer_iters = 500
inner_iters = 1


class Traj(NamedTuple):
    obs: np.ndarray
    acs: np.ndarray
    next_obs: np.ndarray
    rews: np.ndarray
    gammas: np.ndarray


def sample_traj(env, task, fpolicy, params):
    env.reset_task(task)
    obs_buf = np.zeros(
        shape=(TRAJ_LEN, TRAJ_NUM, STATE_DIM),
        dtype=np.float32)
    next_obs_buf = np.zeros(
        shape=(TRAJ_LEN, TRAJ_NUM, STATE_DIM),
        dtype=np.float32)
    acs_buf = np.zeros(
        shape=(TRAJ_LEN, TRAJ_NUM),
        dtype=np.int8)
    rews_buf = np.zeros(shape=(TRAJ_LEN, TRAJ_NUM),
                        dtype=np.float32)
    gammas_buf = np.zeros(shape=(TRAJ_LEN, TRAJ_NUM),
                          dtype=np.float32)
    with torch.no_grad():
        for batch in range(TRAJ_NUM):
            ob = env.reset()
            for step in range(TRAJ_LEN):
                ob_tensor = torch.from_numpy(ob)
                pi, _ = fpolicy(params, ob_tensor)
                ac_tensor = pi.sample()
                ac = ac_tensor.cpu().numpy()
                next_ob, rew, done, info = env.step(ac)

                obs_buf[step][batch] = ob
                next_obs_buf[step][batch] = next_ob
                acs_buf[step][batch] = ac
                rews_buf[step][batch] = rew
                gammas_buf[step][batch] = done * GAMMA
                ob = next_ob
    return Traj(obs=obs_buf, acs=acs_buf, next_obs=next_obs_buf, rews=rews_buf, gammas=gammas_buf)


def a2c_loss(traj, fpolicy, params, value_coef):
    lambdas = np.ones_like(traj.gammas) * LAMBDA
    _, next_values = fpolicy(params, torch.from_numpy(traj.next_obs))
    next_values = torch.squeeze(next_values, -1).detach().numpy()
    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    returns = []
    g = next_values[-1, :]
    for i in reversed(range(next_values.shape[0])):
        g = traj.rews[i, :] + traj.gammas[i, :] * \
            ((1 - lambdas[i, :]) * next_values[i, :] + lambdas[i, :] * g)
        returns.insert(0, g)
    lambda_returns = torch.from_numpy(np.array(returns))
    pi, values = fpolicy(params, torch.from_numpy(traj.obs))
    log_probs = pi.log_prob(torch.from_numpy(traj.acs))
    advs = lambda_returns - torch.squeeze(values, -1)
    action_loss = -(advs.detach() * log_probs).mean()
    value_loss = advs.pow(2).mean()

    a2c_loss = action_loss + value_coef * value_loss
    return a2c_loss


def evaluate(env, seed, task_num, fpolicy, params):
    pre_reward_ls = []
    post_reward_ls = []
    inner_opt = TorchOpt.MetaSGD(lr=0.5)
    env = gym.make('TabularMDP-v0',
                   **dict(num_states=STATE_DIM, num_actions=ACTION_DIM, max_episode_steps=TRAJ_LEN, seed=args.seed))
    tasks = env.sample_tasks(num_tasks=task_num)
    # policy_state_dict = TorchOpt.extract_state_dict(policy)
    # optim_state_dict = TorchOpt.extract_state_dict(inner_opt)
    for idx in range(task_num):
        for _ in range(inner_iters):
            pre_trajs = sample_traj(env, tasks[idx], fpolicy, params)

            inner_loss = a2c_loss(pre_trajs, fpolicy, params, value_coef=0.5)
            params = inner_opt.step(inner_loss, params)
        post_trajs = sample_traj(env, tasks[idx], fpolicy, params)

        # Logging
        pre_reward_ls.append(np.sum(pre_trajs.rews, axis=0).mean())
        post_reward_ls.append(np.sum(post_trajs.rews, axis=0).mean())

        # TorchOpt.recover_state_dict(policy, policy_state_dict)
        # TorchOpt.recover_state_dict(inner_opt, optim_state_dict)
    return pre_reward_ls, post_reward_ls


def main(args):
    # init training
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Env
    env = gym.make('TabularMDP-v0',
                   **dict(num_states=STATE_DIM, num_actions=ACTION_DIM, max_episode_steps=TRAJ_LEN, seed=args.seed))
    # Policy
    policy = CategoricalMLPPolicy(input_size=STATE_DIM,
                                  output_size=ACTION_DIM)
    fpolicy, params = functorch.make_functional(policy)

    inner_opt = TorchOpt.MetaSGD(lr=0.5)
    outer_opt = optim.Adam(params, lr=1e-3)
    train_pre_reward = []
    train_post_reward = []
    test_pre_reward = []
    test_post_reward = []

    for i in range(outer_iters):
        tasks = env.sample_tasks(num_tasks=TASK_NUM)
        train_pre_reward_ls = []
        train_post_reward_ls = []

        outer_opt.zero_grad()

        # policy_state_dict = TorchOpt.extract_state_dict(policy)
        # optim_state_dict = TorchOpt.extract_state_dict(inner_opt)
        param_orig = [p.detach().clone().requires_grad_() for p in params]
        _params = list(params)
        for idx in range(TASK_NUM):

            for _ in range(inner_iters):
                pre_trajs = sample_traj(env, tasks[idx], fpolicy, _params)
                inner_loss = a2c_loss(pre_trajs, fpolicy, _params, value_coef=0.5)
                _params = inner_opt.step(inner_loss, _params)
            post_trajs = sample_traj(env, tasks[idx], fpolicy, _params)
            outer_loss = a2c_loss(post_trajs, fpolicy, _params, value_coef=0.5)
            outer_loss.backward()
            _params = [p.detach().clone().requires_grad_() for p in param_orig]
            # TorchOpt.recover_state_dict(policy, policy_state_dict)
            # TorchOpt.recover_state_dict(inner_opt, optim_state_dict)

            # Logging
            train_pre_reward_ls.append(np.sum(pre_trajs.rews, axis=0).mean())
            train_post_reward_ls.append(np.sum(post_trajs.rews, axis=0).mean())
        outer_opt.step()

        test_pre_reward_ls, test_post_reward_ls = evaluate(
            env, args.seed, TASK_NUM, fpolicy, params)

        train_pre_reward.append(sum(train_pre_reward_ls) / TASK_NUM)
        train_post_reward.append(sum(train_post_reward_ls) / TASK_NUM)
        test_pre_reward.append(sum(test_pre_reward_ls) / TASK_NUM)
        test_post_reward.append(sum(test_post_reward_ls) / TASK_NUM)
        
        print('Train_iters', i)
        print("train_pre_reward", sum(train_pre_reward_ls) / TASK_NUM)
        print("train_post_reward", sum(train_post_reward_ls) / TASK_NUM)
        print("test_pre_reward", sum(test_pre_reward_ls) / TASK_NUM)
        print("test_post_reward", sum(test_post_reward_ls) / TASK_NUM)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                                 'Model-Agnostic Meta-Learning (MAML) - Train')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed (default: 1)')
    args = parser.parse_args()
    main(args)
