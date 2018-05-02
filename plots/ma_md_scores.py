"""Plotting the 8 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(15.0, 8.0))

# wrn, mnist
csv_file = '/data/gilad/logs/ma_scores/wrn/mnist/log_2238_210318_ma_score_wrn_mnist_wd_0.00078-SUPERSEED=21031802/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax1 = fig.add_subplot(331)
ax1.plot(steps, values, 'b')
ax1.set_ylabel('MA score', color='b', labelpad=10, fontdict={'fontsize': 12})
ax1.tick_params('y', colors='b')
ax1.yaxis.grid()
ax1.set_title('MNIST')
ax1.text(-220, 0.95, 'Wide ResNet 28-10', va='center', rotation='vertical', fontdict={'fontsize': 13})

csv_file = '/data/gilad/logs/ma_scores/wrn/mnist/log_2238_210318_ma_score_wrn_mnist_wd_0.00078-SUPERSEED=21031802/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax11 = ax1.twinx()
ax11.plot(steps, values, 'r')
ax11.set_ylim(bottom=0, top=1)
ax11.tick_params('y', colors='r')

# wrn, cifar10
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax2 = fig.add_subplot(332)
ax2.plot(steps, values, 'b')
ax2.tick_params('y', colors='b')
ax2.yaxis.grid()
ax2.set_title('CIFAR-10')

csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = [steps[0]]  + steps[10:]
values = [values[0]] + values[10:]
ax22 = ax2.twinx()
ax22.plot(steps, values, 'r')
ax22.set_ylim(bottom=0, top=1)
ax22.tick_params('y', colors='r')

# wrn, cifar100
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar100/log_1444_070318_wrn_cifar100_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax3 = fig.add_subplot(333)
ax3.plot(steps, values, 'b')
ax3.tick_params('y', colors='b')
ax3.yaxis.grid()
ax3.set_title('CIFAR-100')

csv_file = '/data/gilad/logs/ma_scores/wrn/cifar100/log_1444_070318_wrn_cifar100_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = [steps[0]]  + steps[10:]
values = [values[0]] + values[10:]
ax33 = ax3.twinx()
ax33.plot(steps, values, 'r')
ax33.set_ylim(bottom=0, top=1)
ax33.tick_params('y', colors='r')
ax33.set_ylabel('MD score', color='r', labelpad=5, fontdict={'fontsize': 12})

# lenet, mnist
csv_file = '/data/gilad/logs/ma_scores/lenet/mnist/log_2200_100318_ma_score_lenet_mnist_wd_0.0-SUPERSEED=10031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = [steps[0]]  + steps[10:]
values = [values[0]] + values[10:]
ax4 = fig.add_subplot(334)
ax4.plot(steps, values, 'b')
ax4.set_ylabel('MA score', color='b', labelpad=5, fontdict={'fontsize': 12})
ax4.tick_params('y', colors='b')
ax4.yaxis.grid()
ax4.text(-220, 0.988, 'LeNet', va='center', rotation='vertical', fontdict={'fontsize': 13})

csv_file = '/data/gilad/logs/ma_scores/lenet/mnist/log_2200_100318_ma_score_lenet_mnist_wd_0.0-SUPERSEED=10031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax44 = ax4.twinx()
ax44.plot(steps, values, 'r')
ax44.set_ylim(bottom=0, top=1)
ax44.tick_params('y', colors='r')

# lenet, cifar10
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar10/log_2354_060318_lenet_ma_score_wd_0.008-SUPERSEED=06031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax5 = fig.add_subplot(335)
ax5.plot(steps, values, 'b')
ax5.tick_params('y', colors='b')
ax5.yaxis.grid()

csv_file = '/data/gilad/logs/ma_scores/lenet/cifar10/log_2354_060318_lenet_ma_score_wd_0.008-SUPERSEED=06031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax55 = ax5.twinx()
ax55.plot(steps, values, 'r')
ax55.set_ylim(bottom=0, top=1)
ax55.tick_params('y', colors='r')

# lenet, cifar100
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar100/log_2340_090318_lenet_cifar100_wd_0.01-SUPERSEED=08031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax6 = fig.add_subplot(336)
ax6.plot(steps, values, 'b')
ax6.set_ylim(bottom=0.1, top=1)
ax6.tick_params('y', colors='b')
ax6.yaxis.grid()

csv_file = '/data/gilad/logs/ma_scores/lenet/cifar100/log_2340_090318_lenet_cifar100_wd_0.01-SUPERSEED=08031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax66 = ax6.twinx()
ax66.plot(steps, values, 'r')
ax66.set_ylim(bottom=0, top=1)
ax66.tick_params('y', colors='r')
ax66.set_ylabel('MD score', color='r', labelpad=5, fontdict={'fontsize': 12})

# fc2net, mnist
csv_file = '/data/gilad/logs/ma_scores/fc2net/mnist/log_1409_140318_ma_score_fc2net_mnist_wd_0.0-SUPERSEED=14031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = [steps[0]]  + steps[10:]
values = [values[0]] + values[10:]
ax7 = fig.add_subplot(337)
ax7.plot(steps, values, 'b')
ax7.set_ylim(bottom=0.985, top=1)
ax7.set_ylabel('MA score', color='b', labelpad=5, fontdict={'fontsize': 12})
ax7.tick_params('y', colors='b')
ax7.yaxis.grid()
ax7.text(-220, 0.9925, 'MLP-640', va='center', rotation='vertical', fontdict={'fontsize': 13})

csv_file = '/data/gilad/logs/ma_scores/fc2net/mnist/log_1409_140318_ma_score_fc2net_mnist_wd_0.0-SUPERSEED=14031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax77 = ax7.twinx()
ax77.plot(steps, values, 'r')
ax77.set_ylim(bottom=0, top=1)
ax77.tick_params('y', colors='r')

# fc2net, cifar10
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar10/log_1705_090318_ma_score_fc2net_cifar10_wd_0.0-SUPERSEED=08031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax8 = fig.add_subplot(338)
ax8.plot(steps, values, 'b')
ax8.set_ylim(bottom=0.4, top=1)
ax8.tick_params('y', colors='b')
ax8.yaxis.grid()

csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar10/log_1705_090318_ma_score_fc2net_cifar10_wd_0.0-SUPERSEED=08031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax88 = ax8.twinx()
ax88.plot(steps, values, 'r')
ax88.set_ylim(bottom=0, top=1)
ax88.tick_params('y', colors='r')

# fc2net, cifar100
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar100/log_1353_100318_ma_score_fc2net_cifar100_wd_0.0-SUPERSEED=10031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax9 = fig.add_subplot(339)
ax9.plot(steps, values, 'b')
ax9.set_ylim(bottom=0.4, top=1)
ax9.tick_params('y', colors='b')
ax9.yaxis.grid()

csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar100/log_1353_100318_ma_score_fc2net_cifar100_wd_0.0-SUPERSEED=10031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
ax99 = ax9.twinx()
ax99.plot(steps, values, 'r')
ax99.set_ylim(bottom=0, top=1)
ax99.tick_params('y', colors='r')
ax99.set_ylabel('MD score', color='r', labelpad=5, fontdict={'fontsize': 12})

plt.subplots_adjust(wspace=0.25)
# fig.tight_layout()
plt.savefig('ma_md_scores_vs_iter.png', dpi=350)

