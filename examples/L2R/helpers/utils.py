# Copyright 2022-2023 MetaOPT Team. All Rights Reserved.
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
# https://github.com/uber-research/learning-to-reweight-examples
# ==============================================================================

import random

import numpy as np
import torch
from torch.utils.data import TensorDataset


def get_imbalance_dataset(
    mnist_train,
    mnist_test,
    pos_ratio=0.9,
    ntrain=5000,
    nval=10,
    ntest=500,
    class_0=4,
    class_1=9,
):
    ratio = 1 - pos_ratio
    ratio_test = 0.5

    # In training, we have 10% 4 and 90% 9.
    # In testing, we have 50% 4 and 50% 9.
    x_train = mnist_train.train_data.numpy() / 255.0
    y_train = mnist_train.train_labels.numpy()
    x_test = mnist_test.test_data.numpy() / 255.0
    y_test = mnist_test.test_labels.numpy()
    x_train_0 = x_train[y_train == class_0]
    x_test_0 = x_test[y_test == class_0]

    # First shuffle, negative.
    idx = np.arange(x_train_0.shape[0])
    np.random.shuffle(idx)
    x_train_0 = x_train_0[idx]

    nval_small_neg = int(np.floor(nval * ratio_test))
    ntrain_small_neg = int(np.floor(ntrain * ratio)) - nval_small_neg

    x_val_0 = x_train_0[:nval_small_neg]  # 450 4 in validation.
    x_train_0 = x_train_0[nval_small_neg : nval_small_neg + ntrain_small_neg]  # 500 4 in training.

    print('Number of train negative classes', ntrain_small_neg)
    print('Number of val negative classes', nval_small_neg)

    idx = np.arange(x_test_0.shape[0])
    np.random.shuffle(idx)
    x_test_0 = x_test_0[: int(np.floor(ntest * ratio_test))]  # 450 4 in testing.

    x_train_1 = x_train[y_train == class_1]
    x_test_1 = x_test[y_test == class_1]

    # First shuffle, positive.
    idx = np.arange(x_train_1.shape[0])
    np.random.shuffle(idx)
    x_train_1 = x_train_1[idx]

    nvalsmall_pos = int(np.floor(nval * (1 - ratio_test)))
    ntrainsmall_pos = int(np.floor(ntrain * (1 - ratio))) - nvalsmall_pos

    x_val_1 = x_train_1[:nvalsmall_pos]  # 50 9 in validation.
    x_train_1 = x_train_1[nvalsmall_pos : nvalsmall_pos + ntrainsmall_pos]  # 4500 9 in training.

    idx = np.arange(x_test_1.shape[0])
    np.random.shuffle(idx)
    x_test_1 = x_test_1[idx]
    x_test_1 = x_test_1[: int(np.floor(ntest * (1 - ratio_test)))]  # 500 9 in testing.

    print('Number of train positive classes', ntrainsmall_pos)
    print('Number of val positive classes', nvalsmall_pos)

    y_train_subset = np.concatenate([np.zeros([x_train_0.shape[0]]), np.ones([x_train_1.shape[0]])])
    y_val_subset = np.concatenate([np.zeros([x_val_0.shape[0]]), np.ones([x_val_1.shape[0]])])
    y_test_subset = np.concatenate([np.zeros([x_test_0.shape[0]]), np.ones([x_test_1.shape[0]])])

    x_train_subset = np.concatenate([x_train_0, x_train_1], axis=0)[:, None, :, :]
    x_val_subset = np.concatenate([x_val_0, x_val_1], axis=0)[:, None, :, :]
    x_test_subset = np.concatenate([x_test_0, x_test_1], axis=0)[:, None, :, :]

    # Final shuffle.
    idx = np.arange(x_train_subset.shape[0])
    np.random.shuffle(idx)
    x_train_subset = x_train_subset[idx].astype(np.float32)
    y_train_subset = y_train_subset[idx].astype(np.float32)

    idx = np.arange(x_val_subset.shape[0])
    np.random.shuffle(idx)
    x_val_subset = x_val_subset[idx].astype(np.float32)
    y_val_subset = y_val_subset[idx].astype(np.float32)

    idx = np.arange(x_test_subset.shape[0])
    np.random.shuffle(idx)
    x_test_subset = x_test_subset[idx].astype(np.float32)
    y_test_subset = y_test_subset[idx].astype(np.float32)

    x_train_subset, y_train_subset, x_val_subset, y_val_subset, x_test_subset, y_test_subset = (
        torch.tensor(x_train_subset),
        torch.tensor(y_train_subset),
        torch.tensor(x_val_subset),
        torch.tensor(y_val_subset),
        torch.tensor(x_test_subset),
        torch.tensor(y_test_subset),
    )

    train_set, val_set, test_set = (
        TensorDataset(x_train_subset, y_train_subset),
        TensorDataset(x_val_subset, y_val_subset),
        TensorDataset(x_test_subset, y_test_subset),
    )

    return train_set, val_set, test_set


def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    # Sets the seed for generating random numbers on all GPUs. It's safe to
    # call this function if CUDA is not available; in that case, it is
    # silently ignored.
    torch.cuda.manual_seed_all(seed)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot(baseline, l2r):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style='darkgrid')
    sns.set_theme(style='darkgrid')
    plt.plot(baseline, label='baseline')
    plt.plot(l2r, label='l2r')
    plt.legend()
    plt.ylabel('Test acc')
    plt.xlabel('Epoch')
    plt.title('Comparison between Baseline and L2R')
    plt.savefig('./result.png')
