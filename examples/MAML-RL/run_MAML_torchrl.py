import argparse

import tqdm
import torch
import torch.optim as optim
from torchrl.envs import GymEnv, ParallelEnv
from torchrl.modules import ProbabilisticTDModule, OneHotCategorical
from torchrl.objectives import GAE

import TorchOpt
from helpers.policy import CategoricalMLPPolicy
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


def a2c_loss(traj, policy, value_coef):
    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    gae = GAE(GAMMA, LAMBDA, policy, gradient_mode=True)
    traj = gae(traj)
    # pi, values = policy(torch.from_numpy(traj.obs))
    # log_probs = pi.log_prob(torch.from_numpy(traj.acs))
    # advs = lambda_returns - torch.squeeze(values, -1)
    advantage = traj.get("advantage_diff")
    policy(traj)
    log_probs = traj.get("action_log_prob")
    action_loss = -(advantage.detach() * log_probs).mean()
    value_loss = advantage.pow(2).mean()
    #print(action_loss)
    #print(value_loss)
    #time.sleep(100)
    assert action_loss.requires_grad
    assert value_loss.requires_grad

    a2c_loss = action_loss + value_coef * value_loss
    return a2c_loss


def evaluate(env, dummy_env, seed, task_num, policy):
    pre_reward_ls = []
    post_reward_ls = []
    inner_opt = TorchOpt.MetaSGD(policy, lr=0.5)

    tasks = dummy_env.sample_tasks(num_tasks=task_num)

    policy_state_dict = TorchOpt.extract_state_dict(policy)
    optim_state_dict = TorchOpt.extract_state_dict(inner_opt)
    for idx in range(task_num):
        env.reset_task(tasks[idx])
        for _ in range(inner_iters):
            pre_traj_td = env.rollout(policy, n_steps=TRAJ_LEN)
            inner_loss = a2c_loss(pre_traj_td, policy, value_coef=0.5)
            inner_opt.step(inner_loss)
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
    policy_module = CategoricalMLPPolicy(
        env.observation_spec.shape[-1],
        env.action_spec.shape[-1]
    )
    policy = ProbabilisticTDModule(
        module=policy_module,
        spec=env.action_spec,
        distribution_class=OneHotCategorical,
        return_log_prob=True,
        in_keys=["observation"],
        out_keys=["action", "state_value"],
        default_interaction_mode="random",
    )

    inner_opt = TorchOpt.MetaSGD(policy_module, lr=0.5)
    outer_opt = optim.Adam(policy.parameters(), lr=1e-3)
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

        policy_state_dict = TorchOpt.extract_state_dict(policy)
        optim_state_dict = TorchOpt.extract_state_dict(inner_opt)
        for idx in range(TASK_NUM):
            env.reset_task(tasks[idx])
            for k in range(inner_iters):
                pre_traj_td = env.rollout(policy=policy, n_steps=TRAJ_LEN, auto_reset=True)
                inner_loss = a2c_loss(pre_traj_td, policy, value_coef=0.5)
                inner_opt.step(inner_loss)

            post_traj_td = env.rollout(policy=policy, n_steps=TRAJ_LEN)
            outer_loss = a2c_loss(post_traj_td, policy, value_coef=0.5)
            outer_loss.backward()

            TorchOpt.recover_state_dict(policy, policy_state_dict)
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
                                                           TASK_NUM, policy)

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
