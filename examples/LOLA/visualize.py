#!/usr/bin/env python3

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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot(file):
    data = np.load('result.npy', allow_pickle=True).tolist()
    sns.set(style='darkgrid')
    sns.set_theme(style='darkgrid')
    for step in range(3):
        plt.plot(data[step], label='Step ' + str(step))
    plt.legend()
    plt.xlabel('Iteartions', fontsize=20)
    plt.ylabel('Joint score', fontsize=20)
    plt.savefig('./result.png')


# plot progress:
if __name__ == '__main__':
    plot('result.npy')
