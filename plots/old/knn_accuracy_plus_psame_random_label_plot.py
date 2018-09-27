"""Plotting the 3 KNN accuracy plots for WRN random label plus the psame plots in the same figure"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def set_placeholder(vec, ph=-1, val=1):
    for i in xrange(len(vec)):
        if vec[i] == ph:
            vec[i] = val
    return vec

fig = plt.figure(figsize=(14, 4.5))
# subpos = [0.55, 0.3, 0.4, 0.3]

# KNN ACCURACIES
# wrn, mnist
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___knn_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
ax1 = fig.add_subplot(231)
steps  = steps[0:26]
values = values[0:26]
ax1.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = steps[0:26]
values = values[0:26]
ax1.plot(steps, values, 'b')
ax1.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax1.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
# ax1.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax1.set_ylabel('train accuracy (%)', fontdict={'fontsize': 14})
for item in ax1.get_yticklabels():
    item.set_fontsize(13)
ax1.set_ylim(bottom=-2, top=104)
ax1.set_title('Random labeled MNIST')
ax1.grid(axis='y')
ax1.legend(['k-NN', 'DNN'], loc=(0.02, 0.6), prop={'size': 12})

# wrn, cifar-10
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___knn_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
ax2 = fig.add_subplot(232)
steps  = steps[0:26]
values = values[0:26]
ax2.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = steps[0:26]
values = values[0:26]
ax2.plot(steps, values, 'b')
ax2.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax2.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
# ax2.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax2.set_ylabel('train accuracy (%)', fontdict={'fontsize': 14})
for item in ax2.get_yticklabels():
    item.set_fontsize(13)
ax2.set_ylim(bottom=-2, top=104)
ax2.set_title('Random labeled CIFAR-10')
ax2.grid(axis='y')
ax2.legend(['k-NN', 'DNN'], loc=(0.02, 0.6), prop={'size': 12})

# wrn, cifar-100
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___knn_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
ax3 = fig.add_subplot(233)
steps  = steps[0:26]
values = values[0:26]
ax3.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = steps[0:26]
values = values[0:26]
ax3.plot(steps, values, 'b')
ax3.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax3.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
# ax3.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax3.set_ylabel('train accuracy (%)', fontdict={'fontsize': 14})
for item in ax3.get_yticklabels():
    item.set_fontsize(13)
ax3.set_ylim(bottom=-2, top=104)
ax3.set_title('Random labeled CIFAR-100')
ax3.grid(axis='y')
ax3.legend(['k-NN', 'DNN'], loc=(0.02, 0.6), prop={'size': 12})



# P_SAME
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

ax4 = fig.add_subplot(234)
ax4.plot(steps[0:26], P_SAME_mnist_wrn_train[0:26], 'k--')
ax4.plot(steps[0:26], P_SAME_mnist_wrn[0:26]      , 'k')
ax4.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax4.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax4.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax4.set_ylabel('$P_{SAME}$', labelpad=5, fontdict={'fontsize': 14})
for item in ax4.get_yticklabels():
    item.set_fontsize(13)
ax4.set_ylim(bottom=0.0, top=1.045)
ax4.yaxis.grid()
# ax4.set_title('Random labeled MNIST')
ax4.legend(['train', 'test'], loc=(0.02, 0.6), prop={'size': 12})

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

ax5 = fig.add_subplot(235)
ax5.plot(steps[0:26], P_SAME_cifar10_wrn_train[0:26], 'k--')
ax5.plot(steps[0:26], P_SAME_cifar10_wrn[0:26]      , 'k')
ax5.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax5.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax5.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax5.set_ylabel('$P_{SAME}$', labelpad=5, fontdict={'fontsize': 14})
for item in ax5.get_yticklabels():
    item.set_fontsize(13)
ax5.set_ylim(bottom=0.0, top=1.045)
ax5.yaxis.grid()
# ax5.set_title('Random labeled CIFAR-10')
ax5.legend(['train', 'test'], loc=(0.02, 0.6), prop={'size': 12})

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

ax6 = fig.add_subplot(236)
ax6.plot(steps[0:26], P_SAME_cifar100_wrn_train[0:26], 'k--')
ax6.plot(steps[0:26], P_SAME_cifar100_wrn[0:26]      , 'k')
ax6.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax6.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax6.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax6.set_ylabel('$P_{SAME}$', labelpad=5, fontdict={'fontsize': 14})
for item in ax6.get_yticklabels():
    item.set_fontsize(13)
ax6.set_ylim(bottom=0.0, top=1.045)
ax6.yaxis.grid()
# ax6.set_title('Random labeled CIFAR-100')
ax6.legend(['train', 'test'], loc=(0.02, 0.6), prop={'size': 12})
fig.tight_layout()
plt.subplots_adjust(wspace=0.3)

plt.savefig('knn_dnn_accuracy_plus_psame_random_labels.png', dpi=350)