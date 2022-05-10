import argparse

import tqdm
import torch
import torch.optim as optim
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import ProbabilisticTDModule, OneHotCategorical
from torchrl.objectives import GAE

import TorchOpt
from helpers.policy import ActorCritic
import numpy as np


TASK_NUM = 40
TRAJ_NUM = 20
TRAJ_LEN = 10

NUM_ENVS = TRAJ_NUM  # number of envs to run in parallel

STATE_DIM = 10
ACTION_DIM = 5

GAMMA = 0.99
LAMBDA = 0.95

outer_iters = 500
inner_iters = 1


def a2c_loss(traj, policy, value, value_coef):
    dist, *_ = policy.get_dist(traj)
    action = traj.get("action")
    log_probs = dist.log_prob(action)

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    gae = GAE(GAMMA, LAMBDA, value, gradient_mode=True)
    traj = gae(traj)
    advantage = traj.get("advantage")
    value_target = traj.get("value_target")
    action_loss = -(advantage * log_probs.view_as(advantage)).mean()
    value_loss = value_target.pow(2).mean()
    assert action_loss.requires_grad
    assert not advantage.requires_grad
    assert not action.requires_grad
    assert value_loss.requires_grad

    loss = action_loss + value_coef * value_loss
    return loss


def evaluate(env, dummy_env, seed, task_num, actor_critic, policy, value):
    pre_reward_ls = []
    post_reward_ls = []
    inner_opt = TorchOpt.MetaSGD(actor_critic, lr=0.05)

    tasks = dummy_env.sample_tasks(num_tasks=task_num)

    policy_state_dict = TorchOpt.extract_state_dict(policy)
    optim_state_dict = TorchOpt.extract_state_dict(inner_opt)
    for idx in range(task_num):
        env.reset_task(tasks[idx])
        for _ in range(inner_iters):
            with set_exploration_mode("random"), torch.no_grad():
                pre_traj_td = env.rollout(policy, n_steps=TRAJ_LEN)
            inner_loss = a2c_loss(pre_traj_td, policy, value, value_coef=0.5)
            inner_opt.step(inner_loss)
        with set_exploration_mode("random"), torch.no_grad():
            post_traj_td = env.rollout(policy, n_steps=TRAJ_LEN)

        # Logging

        pre_reward_ls.append(torch.sum(pre_traj_td.get("reward"),dim=1).mean().item())
        post_reward_ls.append(torch.sum(post_traj_td.get("reward"),dim=1).mean().item())

        TorchOpt.recover_state_dict(policy, policy_state_dict)
        TorchOpt.recover_state_dict(inner_opt, optim_state_dict)
    return pre_reward_ls, post_reward_ls


def main(args):
    # init training
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Env
    lambda_env = lambda: GymEnv(
        "TabularMDP-v0",
        num_states=STATE_DIM,
        num_actions=ACTION_DIM,
        max_episode_steps=TRAJ_LEN,
        seed=args.seed,
    )
    if args.parallel:
        env = ParallelEnv(NUM_ENVS, lambda_env)
    else:
        env = lambda_env()
    env.reset()
    # Policy
    obs_key = list(env.observation_spec.keys())[0]
    actor_critic = ActorCritic(
        env.observation_spec[obs_key].shape[-1],
        env.action_spec.shape[-1]
    )
    policy = actor_critic.get_policy_operator()
    value = actor_critic.get_value_operator()

    inner_opt = TorchOpt.MetaSGD(actor_critic, lr=0.05)
    outer_opt = optim.Adam(actor_critic.parameters(), lr=1e-3)
    train_pre_reward = []
    train_post_reward = []
    test_pre_reward = []
    test_post_reward = []

    dummy_env = lambda_env()

    pbar = tqdm.tqdm(range(outer_iters))
    for i in pbar:
        tasks = dummy_env.sample_tasks(num_tasks=TASK_NUM)
        train_pre_reward_ls = []
        train_post_reward_ls = []

        outer_opt.zero_grad()

        policy_state_dict = TorchOpt.extract_state_dict(actor_critic)
        optim_state_dict = TorchOpt.extract_state_dict(inner_opt)
        for idx in range(TASK_NUM):
            env.reset_task(tasks[idx])
            for k in range(inner_iters):
                with set_exploration_mode("random"), torch.no_grad():
                    pre_traj_td = env.rollout(policy=policy, n_steps=TRAJ_LEN, auto_reset=True)
                for k, item in pre_traj_td.items():
                    # print(k, item.requires_grad)
                    if item.requires_grad:
                        raise RuntimeError
                # print("\n\n")
                inner_loss = a2c_loss(pre_traj_td, policy, value, value_coef=0.5)
                # for k, item in pre_traj_td.items():
                #     print(k, item.requires_grad)
                # print("\n\n")
                inner_opt.step(inner_loss)

            with set_exploration_mode("random"), torch.no_grad():
                post_traj_td = env.rollout(policy=policy, n_steps=TRAJ_LEN)
            outer_loss = a2c_loss(post_traj_td, policy, value, value_coef=0.5)
            outer_loss.backward()

            TorchOpt.recover_state_dict(actor_critic, policy_state_dict)
            TorchOpt.recover_state_dict(inner_opt, optim_state_dict)

            # Logging
            #print(pre_traj_td.get("reward").size())
            #print(torch.sum(pre_traj_td.get("reward").item(),dim=1))
            #time.sleep(100)
            train_pre_reward_ls.append(torch.sum(pre_traj_td.get("reward"),dim=1).mean().item())
            #print(pre_traj_td.get("reward").sum(dim=0).item())
            train_post_reward_ls.append(torch.sum(post_traj_td.get("reward"),dim=1).mean().item())

        outer_opt.step()

        test_pre_reward_ls, test_post_reward_ls = evaluate(env, dummy_env, args.seed,
                                                           TASK_NUM, actor_critic,
                                                           policy, value)

        train_pre_reward.append(sum(train_pre_reward_ls) / TASK_NUM)
        train_post_reward.append(sum(train_post_reward_ls) / TASK_NUM)
        test_pre_reward.append(sum(test_pre_reward_ls) / TASK_NUM)
        test_post_reward.append(sum(test_post_reward_ls) / TASK_NUM)
        pbar.set_description(f"train_pre_reward: {train_pre_reward[-1]: 4.4f}, "
                             f"train_post_reward: {train_post_reward[-1]: 4.4f}, "
                             f"test_pre_reward: {test_pre_reward[-1]: 4.4f}, "
                             f"test_post_reward: {test_post_reward[-1]: 4.4f}, "
                             )
        np.save("train_pre_reward_{}.npy".format(args.seed), np.array(train_pre_reward))
        np.save("train_post_reward_{}.npy".format(args.seed), np.array(train_post_reward))
        np.save("test_pre_reward_{}.npy".format(args.seed), np.array(test_pre_reward))
        np.save("test_post_reward_{}.npy".format(args.seed), np.array(test_post_reward))

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                                 'Model-Agnostic Meta-Learning (MAML) - Train')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--parallel',
                        action='store_true',
                        help="run envs in parallel")
    args = parser.parse_args()
    main(args)
