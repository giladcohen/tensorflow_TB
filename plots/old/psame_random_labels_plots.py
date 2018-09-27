"""Plotting the 3 P_SAME plots for MNIST, CIFAR-10 and CIFAR-100. Every subplot has 2 plots for the train set and the test set."""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def set_placeholder(vec, ph=-1, val=1):
    for i in xrange(len(vec)):
        if vec[i] == ph:
            vec[i] = val
    return vec

subpos = [0.4, 0.5, 0.48, 0.3]
fig = plt.figure(figsize=(14, 4))

# wrn, mnist
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___ma_score_trainset'
steps, ma_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___md_score_trainset'
steps, md_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
md_values_train = set_placeholder(md_values_train)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, acc_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_mnist_wrn_train = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values_train, md_values_train, acc_values_train)]
P_SAME_mnist_wrn_train = [round(elem, 4) for elem in P_SAME_mnist_wrn_train]

csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_mnist_wrn= [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_mnist_wrn = [round(elem, 4) for elem in P_SAME_mnist_wrn]

ax1 = fig.add_subplot(131)
ax1.plot(steps[0:26], P_SAME_mnist_wrn_train[0:26], 'k--')
ax1.plot(steps[0:26], P_SAME_mnist_wrn[0:26]      , 'k')
ax1.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax1.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax1.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax1.set_ylabel('$P_{SAME}$', labelpad=5, fontdict={'fontsize': 12})
ax1.set_ylim(bottom=0.0, top=1.045)
ax1.yaxis.grid()
ax1.set_title('Random labeled MNIST')
ax1.legend(['train', 'test'], loc=(0.05, 0.76), prop={'size': 12})


# wrn, cifar10
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score_trainset'
steps, ma_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score_trainset'
steps, md_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
md_values_train = set_placeholder(md_values_train)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, acc_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar10_wrn_train = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values_train, md_values_train, acc_values_train)]
P_SAME_cifar10_wrn_train = [round(elem, 4) for elem in P_SAME_cifar10_wrn_train]

csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar10_wrn= [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_cifar10_wrn = [round(elem, 4) for elem in P_SAME_cifar10_wrn]

ax2 = fig.add_subplot(132)
ax2.plot(steps[0:26], P_SAME_cifar10_wrn_train[0:26], 'k--')
ax2.plot(steps[0:26], P_SAME_cifar10_wrn[0:26]      , 'k')
ax2.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax2.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax2.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax2.set_ylabel('$P_{SAME}$', labelpad=5, fontdict={'fontsize': 12})
ax2.set_ylim(bottom=0.0, top=1.045)
ax2.yaxis.grid()
ax2.set_title('Random labeled CIFAR-10')
ax2.legend(['train', 'test'], loc=(0.05, 0.76), prop={'size': 12})


# wrn, cifar100
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score_trainset'
steps, ma_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score_trainset'
steps, md_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
md_values_train = set_placeholder(md_values_train)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, acc_values_train = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar100_wrn_train = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values_train, md_values_train, acc_values_train)]
P_SAME_cifar100_wrn_train = [round(elem, 4) for elem in P_SAME_cifar100_wrn_train]

csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar100_wrn= [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_cifar100_wrn = [round(elem, 4) for elem in P_SAME_cifar100_wrn]

ax3 = fig.add_subplot(133)
ax3.plot(steps[0:26], P_SAME_cifar100_wrn_train[0:26], 'k--')
ax3.plot(steps[0:26], P_SAME_cifar100_wrn[0:26]      , 'k')
ax3.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax3.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax3.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax3.set_ylabel('$P_{SAME}$', labelpad=5, fontdict={'fontsize': 12})
ax3.set_ylim(bottom=0.0, top=1.045)
ax3.yaxis.grid()
ax3.set_title('Random labeled CIFAR-100')
ax3.legend(['train', 'test'], loc=(0.05, 0.76), prop={'size': 12})


fig.tight_layout()
plt.savefig('psame_scores_random_labels_vs_iter.png', dpi=350)
