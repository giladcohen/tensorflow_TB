from utils.plots import load_data_from_csv_wrapper
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(14.0, 8.0))

# wrn, cifar10
# MA
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:21]
values = values[0:21]
ax1 = fig.add_subplot(121)
ax1.plot(steps, values, 'b--')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:21]
values = values[0:21]
ax1.plot(steps, values, 'b')
ax1.set_xticks([0, 5000, 10000, 15000, 20000])
ax1.set_xticklabels(['0', '5', '10', '15', '20'], fontdict={'fontsize': 13})
ax1.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax1.yaxis.grid()
ax1.set_ylabel('MA score', color='b', labelpad=5, fontdict={'fontsize': 14})
for item in ax1.get_yticklabels():
    item.set_fontsize(13)
ax1.set_title('Random Labeled CIFAR-10')
ax1.legend(['train', 'test'], loc=(0.03, 0.88), prop={'size': 16})

# MD
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:21]
values = values[0:21]
ax2 = fig.add_subplot(122)
ax2.plot(steps, values, 'r--')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:21]
values = values[0:21]
ax2.plot(steps, values, 'r')
ax2.set_xticks([0, 5000, 10000, 15000, 20000])
ax2.set_xticklabels(['0', '5', '10', '15', '20'], fontdict={'fontsize': 13})
ax2.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax2.yaxis.grid()
ax2.set_ylabel('MD score', color='r', labelpad=5, fontdict={'fontsize': 14})
for item in ax2.get_yticklabels():
    item.set_fontsize(13)
ax2.set_title('Random Labeled CIFAR-10')
ax2.legend(['train', 'test'], loc=(0.03, 0.88), prop={'size': 16})

fig.tight_layout()
plt.savefig('ma_md_scores_trainset_just_cifar10_vs_iter.png', dpi=350)
