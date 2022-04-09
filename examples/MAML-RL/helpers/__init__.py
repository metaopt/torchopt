# This file is modified from:
# https://github.com/tristandeleu/pytorch-maml-rl
from gym.envs.registration import register

register(
    'TabularMDP-v0',
    entry_point='helpers.Tabular_mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5, 'max_episode_steps':10, 'seed':1}
    )




 