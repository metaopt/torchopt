import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot(file):
    data = np.load('result.npy', allow_pickle=True).tolist()
    sns.set(style='darkgrid')
    sns.set_theme(style="darkgrid")
    for step in range(3):
        plt.plot(data[step], label='Step ' + str(step))
    plt.legend()
    plt.xlabel('Iteartions', fontsize=20)
    plt.ylabel('Joint score', fontsize=20)
    plt.savefig('./result.png')


# plot progress:
if __name__ == "__main__":
    plot('result.npy')
