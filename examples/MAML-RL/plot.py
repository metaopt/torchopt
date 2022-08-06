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

# coding: utf-8

# In[3]:


import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


seed_list = [1, 2, 3, 4, 5]


for seed in seed_list:
    if seed == 1:
        train_pre_reward = np.expand_dims(np.load('train_pre_reward_{}.npy'.format(seed)), axis=0)
        train_post_reward = np.expand_dims(np.load('train_post_reward_{}.npy'.format(seed)), axis=0)
        test_pre_reward = np.expand_dims(np.load('test_pre_reward_{}.npy'.format(seed)), axis=0)
        test_post_reward = np.expand_dims(np.load('test_post_reward_{}.npy'.format(seed)), axis=0)
    else:
        train_pre_reward = np.concatenate(
            [
                train_pre_reward,
                np.expand_dims(np.load('train_pre_reward_{}.npy'.format(seed)), axis=0),
            ],
            axis=0,
        )
        train_post_reward = np.concatenate(
            [
                train_post_reward,
                np.expand_dims(np.load('train_post_reward_{}.npy'.format(seed)), axis=0),
            ],
            axis=0,
        )
        test_pre_reward = np.concatenate(
            [
                test_pre_reward,
                np.expand_dims(np.load('test_pre_reward_{}.npy'.format(seed)), axis=0),
            ],
            axis=0,
        )
        test_post_reward = np.concatenate(
            [
                test_post_reward,
                np.expand_dims(np.load('test_post_reward_{}.npy'.format(seed)), axis=0),
            ],
            axis=0,
        )
sns.set(style='darkgrid')
sns.set_theme(style='darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1)
train_pre_mean = np.mean(train_pre_reward[:], axis=0)
train_pre_std = np.std(train_pre_reward[:], axis=0)
ax1.plot(train_pre_mean, label='Train-Pre')
ax1.fill_between(
    np.arange(500), train_pre_mean - train_pre_std, train_pre_mean + train_pre_std, alpha=0.2
)
train_post_mean = np.mean(train_post_reward[:], axis=0)
train_post_std = np.std(train_post_reward[:], axis=0)
ax1.plot(train_post_mean, label='Train-Post')
ax1.legend()
ax1.set_title('Train')
ax1.fill_between(
    np.arange(500), train_post_mean - train_post_std, train_post_mean + train_post_std, alpha=0.2
)
# ax1.set_xlabel('Itr')
ax1.set_ylabel('Reward')


test_pre_mean = np.mean(test_pre_reward[:], axis=0)
test_pre_std = np.std(test_pre_reward[:], axis=0)
ax2.plot(test_pre_mean, label='Test-Pre')
ax2.fill_between(
    np.arange(500), test_pre_mean - test_pre_std, test_pre_mean + test_pre_std, alpha=0.2
)
test_post_mean = np.mean(test_post_reward[:], axis=0)
test_post_std = np.std(test_post_reward[:], axis=0)
ax2.plot(test_post_mean, label='Test-Post')
ax2.fill_between(
    np.arange(500), test_post_mean - test_post_std, test_post_mean + test_post_std, alpha=0.2
)
ax2.set_xlabel('Itrerations')
ax2.set_ylabel('Reward')
ax2.set_title('Test')
ax2.legend()
#
# fig.legend(ncol=2, loc='lower right')

# fig.savefig('MAML.png')
plt.tight_layout()
plt.savefig('./maml.png', bbox_inches='tight', pad_inches=0.04, dpi=150)
plt.show()
