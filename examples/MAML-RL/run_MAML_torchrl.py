import argparse

import tqdm
import torch
import torch.optim as optim
from torchrl.envs import GymEnv, ParallelEnv, SerialEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import ProbabilisticTDModule, OneHotCategorical
from torchrl.objectives.returns.functional import vec_td_lambda_advantage_estimate
from torchrl.envs.utils import step_tensordict

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


def a2c_loss(traj, policy_module, value_module, value_coef):
    dist, *_ = policy_module.get_dist(traj)
    action = traj.get("action")
    log_probs = dist.log_prob(action)
    value_module.module[-1](traj)  # will read the "hidden" key and return a state value
    value = traj.get("state_value")

    reward = traj.get("reward")
    done = traj.get("done")
    next_traj = step_tensordict(traj)
    next_value = value_module(next_traj).get("state_value")

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    # tderror = TDEstimate(GAMMA, value_module, gradient_mode=True)
    # tderror = TDLambdaEstimate(GAMMA, LAMBDA, value_module, gradient_mode=True)
    with torch.no_grad():
        advantage = vec_td_lambda_advantage_estimate(
            GAMMA, LAMBDA, value, next_value, reward, done
        )
    value_error = value - (advantage + value).detach()
    action_loss = -(advantage * log_probs.view_as(advantage)).mean()
    value_loss = value_error.pow(2).mean()

    assert action_loss.requires_grad
    assert not advantage.requires_grad
    assert not action.requires_grad
    assert value_loss.requires_grad

    loss = action_loss + value_coef * value_loss
    return loss


def evaluate(env, dummy_env, seed, task_num, actor_critic, policy, value):
    pre_reward_ls = []
    post_reward_ls = []
    device = next(actor_critic.parameters()).device

    inner_opt = TorchOpt.MetaSGD(actor_critic, lr=0.5)

    tasks = dummy_env.sample_tasks(num_tasks=task_num)

    policy_state_dict = TorchOpt.extract_state_dict(actor_critic)
    optim_state_dict = TorchOpt.extract_state_dict(inner_opt)
    for idx in range(task_num):
        env.reset_task(tasks[idx])
        for _ in range(inner_iters):
            with set_exploration_mode("random"), torch.no_grad():
                pre_traj_td = env.rollout(policy, n_steps=TRAJ_LEN).to(device)
            inner_loss = a2c_loss(pre_traj_td, policy, value, value_coef=0.5)
            inner_opt.step(inner_loss)
        with set_exploration_mode("random"), torch.no_grad():
            post_traj_td = env.rollout(policy, n_steps=TRAJ_LEN).to(device)

        # Logging
        pre_reward_ls.append(torch.sum(pre_traj_td.get("reward"),dim=1).mean().item())
        post_reward_ls.append(torch.sum(post_traj_td.get("reward"),dim=1).mean().item())

        TorchOpt.recover_state_dict(actor_critic, policy_state_dict)
        TorchOpt.recover_state_dict(inner_opt, optim_state_dict)
    return pre_reward_ls, post_reward_ls


def main(args):
    device = 'cuda:0' if torch.cuda.device_count() else 'cpu'
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
        env = SerialEnv(NUM_ENVS, lambda_env)
    env.reset()
    # Policy
    obs_key = list(env.observation_spec.keys())[0]
    actor_critic_module = ActorCritic(
        env.observation_spec[obs_key].shape[-1],
        env.action_spec.shape[-1]
    ).to(device)
    policy_module = actor_critic_module.get_policy_operator()
    value_module = actor_critic_module.get_value_operator()

    inner_opt = TorchOpt.MetaSGD(actor_critic_module, lr=0.5)
    outer_opt = optim.Adam(actor_critic_module.parameters(), lr=1e-3)
    train_pre_reward = []
    train_post_reward = []
    test_pre_reward = []
    test_post_reward = []

    dummy_env = lambda_env()

    pbar = tqdm.tqdm(range(outer_iters))
    for i in pbar:
        # print("i: ", i)
        tasks = dummy_env.sample_tasks(num_tasks=TASK_NUM)
        train_pre_reward_ls = []
        train_post_reward_ls = []

        outer_opt.zero_grad()
        policy_state_dict = TorchOpt.extract_state_dict(actor_critic_module)
        optim_state_dict = TorchOpt.extract_state_dict(inner_opt)
        for idx in range(TASK_NUM):
            # print("idx: ", idx)
            env.reset_task(tasks[idx])
            for k in range(inner_iters):
                with set_exploration_mode("random"), torch.no_grad():
                    pre_traj_td = env.rollout(policy=policy_module, n_steps=TRAJ_LEN, auto_reset=True).to(device)
                inner_loss = a2c_loss(pre_traj_td, policy_module, value_module, value_coef=0.5)
                inner_opt.step(inner_loss)

            with set_exploration_mode("random"), torch.no_grad():
                post_traj_td = env.rollout(policy=policy_module, n_steps=TRAJ_LEN).to(device)
            outer_loss = a2c_loss(post_traj_td, policy_module, value_module, value_coef=0.5)
            outer_loss.backward()

            TorchOpt.recover_state_dict(actor_critic_module, policy_state_dict)
            TorchOpt.recover_state_dict(inner_opt, optim_state_dict)

            # Logging
            train_pre_reward_ls.append(torch.sum(pre_traj_td.get("reward"),dim=1).mean().item())
            train_post_reward_ls.append(torch.sum(post_traj_td.get("reward"),dim=1).mean().item())

        outer_opt.step()

        test_pre_reward_ls, test_post_reward_ls = evaluate(env, dummy_env, args.seed,
                                                           TASK_NUM, actor_critic_module,
                                                           policy_module, value_module)

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

