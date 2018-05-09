from utils.plots import load_data_from_csv_wrapper
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def set_placeholder(vec, ph=-1, val=1):
    for i in xrange(len(vec)):
        if vec[i] == ph:
            vec[i] = val
    return vec


fig = plt.figure(figsize=(14, 6))

# wrn, mnist
# MA
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___ma_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax1 = fig.add_subplot(231)
ax1.plot(steps, values, 'b--')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax1.plot(steps, values, 'b')
ax1.set_xticklabels([], fontdict={'fontsize': 13})
ax1.yaxis.grid()
ax1.set_ylabel('$MC$ score', color='b', labelpad=5, fontdict={'fontsize': 14})
ax1.set_ylim(bottom=-0.04, top=1.04)
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
for item in ax1.get_yticklabels():
    item.set_fontsize(13)
ax1.set_title('Random labeled MNIST', fontdict={'fontsize': 15})
ax1.legend(['train', 'test'], loc=(0.72, 0.4), prop={'size': 12})

# MD
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___md_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
values = set_placeholder(values)
steps  = steps[0:26]
values = values[0:26]
ax2 = fig.add_subplot(234)
ax2.plot(steps, values, 'r--')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/mnist/log_0103_300318_wrn_mnist_wd_0_no_aug_steps_50k-SUPERSEED=30031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax2.plot(steps, values, 'r')
# ax2.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
ax2.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax2.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax2.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax2.yaxis.grid()
ax2.set_ylabel('$ME$ score', color='r', labelpad=5, fontdict={'fontsize': 14})
ax2.set_ylim(bottom=-0.04, top=1.04)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
for item in ax2.get_yticklabels():
    item.set_fontsize(13)
ax2.legend(['train', 'test'], loc=(0.72, 0.4), prop={'size': 12})

# wrn, cifar10
# MA
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax3 = fig.add_subplot(232)
ax3.plot(steps, values, 'b--')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax3.plot(steps, values, 'b')
ax3.set_xticklabels([], fontdict={'fontsize': 13})
ax3.yaxis.grid()
ax3.set_ylim(bottom=-0.04, top=1.04)
ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
for item in ax3.get_yticklabels():
    item.set_fontsize(13)
ax3.set_title('Random labeled CIFAR-10', fontdict={'fontsize': 15})
ax3.legend(['train', 'test'], loc=(0.72, 0.4), prop={'size': 12})

# MD
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
values = set_placeholder(values)
steps  = steps[0:26]
values = values[0:26]
ax4 = fig.add_subplot(235)
ax4.plot(steps, values, 'r--')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar10/log_0103_300318_wrn_cifar10_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax4.plot(steps, values, 'r')
ax4.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax4.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax4.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax4.yaxis.grid()
ax4.set_ylim(bottom=-0.04, top=1.04)
ax4.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
for item in ax4.get_yticklabels():
    item.set_fontsize(13)
ax4.legend(['train', 'test'], loc=(0.72, 0.4), prop={'size': 12})

# wrn, cifar100
# MA
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax5 = fig.add_subplot(233)
ax5.plot(steps, values, 'b--')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax5.plot(steps, values, 'b')
ax5.set_xticklabels([], fontdict={'fontsize': 13})
ax5.yaxis.grid()
ax5.set_ylim(bottom=-0.04, top=1.04)
ax5.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
for item in ax5.get_yticklabels():
    item.set_fontsize(13)
ax5.set_title('Random labeled CIFAR-100', fontdict={'fontsize': 15})
ax5.legend(['train', 'test'], loc=(0.72, 0.4), prop={'size': 12})

# MD
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score_trainset'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
values = set_placeholder(values)
steps  = steps[0:26]
values = values[0:26]
ax6 = fig.add_subplot(236)
ax6.plot(steps, values, 'r--')
csv_file = '/data/gilad/logs/ma_scores/random_labels/wrn/cifar100/log_0103_300318_wrn_cifar100_wd_0_no_aug-SUPERSEED=30031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = steps[0:26]
values = values[0:26]
ax6.plot(steps, values, 'r')
ax6.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
ax6.set_xticklabels(['0', '5', '10', '15', '20', '25'], fontdict={'fontsize': 13})
ax6.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax6.yaxis.grid()
ax6.set_ylim(bottom=-0.04, top=1.04)
ax6.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
for item in ax6.get_yticklabels():
    item.set_fontsize(13)
ax6.legend(['train', 'test'], loc=(0.72, 0.4), prop={'size': 12})

fig.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.savefig('ma_md_scores_trainset_vs_iter_25k.png', dpi=350)

