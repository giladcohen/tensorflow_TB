"""Plotting the 8 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(10, 8.0))
# subpos = [0.55, 0.3, 0.4, 0.3]

# wrn, cifar10
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___knn_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = steps[0:21]
values = values[0:21]
ax2 = fig.add_subplot(111)
ax2.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = steps[0:21]
values = values[0:21]
ax2.plot(steps, values, 'black')
ax2.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000])
ax2.set_xticklabels(['0', '2.5', '5', '7.5', '10', '12.5', '15', '17.5', '20'], fontdict={'fontsize': 13})
ax2.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax2.set_ylabel('train accuracy (%)', fontdict={'fontsize': 14})
for item in ax2.get_yticklabels():
    item.set_fontsize(13)
ax2.legend(['k-NN', 'DNN'], loc=(0.05, 0.86), prop={'size': 16})
ax2.set_title('Random Labeled CIFAR-10')
ax2.grid(axis='y')
plt.savefig('knn_dnn_accuracy_random_labels_just_cifar10_vs_iter.png', dpi=350)


