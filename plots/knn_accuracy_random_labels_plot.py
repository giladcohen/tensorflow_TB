"""Plotting the 8 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(15, 8.0))
# subpos = [0.55, 0.3, 0.4, 0.3]

# wrn, mnist
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___knn_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
ax1 = fig.add_subplot(131)
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
ax1.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax1.set_ylabel('train accuracy (%)', fontdict={'fontsize': 14})
for item in ax1.get_yticklabels():
    item.set_fontsize(13)
ax1.set_ylim(bottom=-2, top=102)
ax1.set_title('Random Labeled MNIST')
ax1.grid(axis='y')
ax1.legend(['k-NN', 'DNN'], loc=(0.05, 0.86), prop={'size': 16})

# wrn, cifar-10
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___knn_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
ax2 = fig.add_subplot(132)
steps  = steps[0:21]
values = values[0:21]
ax2.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = steps[0:21]
values = values[0:21]
ax2.plot(steps, values, 'b')
ax2.set_xticks([0, 5000, 10000, 15000, 20000])
ax2.set_xticklabels(['0', '5', '10', '15', '20'], fontdict={'fontsize': 13})
ax2.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
for item in ax2.get_yticklabels():
    item.set_fontsize(13)
ax2.set_ylim(bottom=-2, top=102)
ax2.set_title('Random Labeled CIFAR-10')
ax2.grid(axis='y')
ax2.legend(['k-NN', 'DNN'], loc=(0.05, 0.86), prop={'size': 16})

# wrn, cifar-100
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___knn_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
ax3 = fig.add_subplot(133)
steps  = steps[0:16]
values = values[0:16]
ax3.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = steps[0:16]
values = values[0:16]
ax3.plot(steps, values, 'b')
ax3.set_xticks([0, 5000, 10000, 15000])
ax3.set_xticklabels(['0', '5', '10', '15'], fontdict={'fontsize': 13})
ax3.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
for item in ax3.get_yticklabels():
    item.set_fontsize(13)
ax3.set_ylim(bottom=-2, top=102)
ax3.set_title('Random Labeled CIFAR-100')
ax3.grid(axis='y')
ax3.legend(['k-NN', 'DNN'], loc=(0.05, 0.86), prop={'size': 16})

fig.tight_layout()
plt.savefig('knn_dnn_accuracy_random_labels_vs_iter.png', dpi=350)


