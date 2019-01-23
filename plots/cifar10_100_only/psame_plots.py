"""Plotting the 3 P_SAME plots for MNIST, CIFAR-10 and CIFAR-100. Every subplot has 3 plots for the 3 architectures."""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

subpos = [0.4, 0.5, 0.48, 0.3]
fig = plt.figure(figsize=(8, 2.8))

# cifar10
# wrn
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar10_wrn = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_cifar10_wrn = [round(elem, 4) for elem in P_SAME_cifar10_wrn]

# lenet
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar10/log_2354_060318_lenet_ma_score_wd_0.008-SUPERSEED=06031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar10/log_2354_060318_lenet_ma_score_wd_0.008-SUPERSEED=06031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar10/log_2354_060318_lenet_ma_score_wd_0.008-SUPERSEED=06031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar10_lenet = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_cifar10_lenet = [round(elem, 4) for elem in P_SAME_cifar10_lenet]

# fc2net
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar10/log_1705_090318_ma_score_fc2net_cifar10_wd_0.0-SUPERSEED=08031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar10/log_1705_090318_ma_score_fc2net_cifar10_wd_0.0-SUPERSEED=08031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar10/log_1705_090318_ma_score_fc2net_cifar10_wd_0.0-SUPERSEED=08031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar10_fc2net = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_cifar10_fc2net = [round(elem, 4) for elem in P_SAME_cifar10_fc2net]

ax2 = fig.add_subplot(121)
ax2.plot(steps, P_SAME_cifar10_wrn   , 'r')
ax2.plot(steps, P_SAME_cifar10_lenet , 'g')
ax2.plot(steps, P_SAME_cifar10_fc2net, 'b')

ax2.set_ylabel('$P_{SAME}$', labelpad=5, fontdict={'fontsize': 15})
ax2.set_ylim(bottom=0.0, top=1.045)
ax2.yaxis.grid()
ax2.set_title('CIFAR-10', fontdict={'fontsize': 12})
ax2.legend(['Wide Resnet', 'LeNet', 'MLP-640'], loc=(0.5, 0.2), prop={'size': 11})

# cifar100
# wrn
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar100/log_1444_070318_wrn_cifar100_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar100/log_1444_070318_wrn_cifar100_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar100/log_1444_070318_wrn_cifar100_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar100_wrn = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_cifar100_wrn = [round(elem, 4) for elem in P_SAME_cifar100_wrn]

# lenet
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar100/log_2340_090318_lenet_cifar100_wd_0.01-SUPERSEED=08031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar100/log_2340_090318_lenet_cifar100_wd_0.01-SUPERSEED=08031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar100/log_2340_090318_lenet_cifar100_wd_0.01-SUPERSEED=08031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar100_lenet = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_cifar100_lenet = [round(elem, 4) for elem in P_SAME_cifar100_lenet]

# fc2net
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar100/log_1353_100318_ma_score_fc2net_cifar100_wd_0.0-SUPERSEED=10031800/data_for_figures/test___ma_score'
steps, ma_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar100/log_1353_100318_ma_score_fc2net_cifar100_wd_0.0-SUPERSEED=10031800/data_for_figures/test___md_score'
steps, md_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar100/log_1353_100318_ma_score_fc2net_cifar100_wd_0.0-SUPERSEED=10031800/data_for_figures/test___score'
steps, acc_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
P_SAME_cifar100_fc2net = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_values, md_values, acc_values)]
P_SAME_cifar100_fc2net = [round(elem, 4) for elem in P_SAME_cifar100_fc2net]

ax3 = fig.add_subplot(122)
ax3.plot(steps, P_SAME_cifar100_wrn   , 'r')
ax3.plot(steps, P_SAME_cifar100_lenet , 'g')
ax3.plot(steps, P_SAME_cifar100_fc2net, 'b')

ax3.set_ylabel('$P_{SAME}$', labelpad=5, fontdict={'fontsize': 15})
ax3.set_ylim(bottom=0.0, top=1.045)
ax3.yaxis.grid()
ax3.set_title('CIFAR-100', fontdict={'fontsize': 12})
ax3.legend(['Wide Resnet', 'LeNet', 'MLP-640'], loc=(0.5, 0.4), prop={'size': 11})

# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=None)

# # collect again just the step values
# csv_file = '/data/gilad/logs/ma_scores/wrn/mnist/log_0746_020518_ma_score_wrn_mnist_wd_0.00078_steps_1000-SUPERSEED=21031802/data_for_figures/test___ma_score'
# steps, _ = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
#
# subax1 = add_subplot_axes(ax1, subpos)
# subax1.set_ylim([0.995, 1.0])
# subax1.set_yticks([0.995, 1.0])
# subax1.plot(steps[-21:], P_SAME_mnist_wrn[-21:]  ,  'r')
# subax1.plot(steps[-21:], P_SAME_mnist_lenet[-21:],  'g')
# subax1.plot(steps[-21:], P_SAME_mnist_fc2net[-21:], 'b')
# ax1.add_patch(patches.Polygon(xy=np.array([[800, 1.03], [390, 0.838], [918, 0.838], [918, 0.52], [1000, 0.97]]), closed=True, color='silver'))
# ax1.add_patch(patches.Rectangle(xy=(800, 0.97), width=200, height=0.06, facecolor='moccasin'))

fig.tight_layout()
plt.savefig('psame_scores_vs_iter.png', dpi=350)
