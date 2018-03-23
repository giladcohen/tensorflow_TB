"""Plotting the 8 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ax1 = fig.add_subplot(331)                          # wrn, mnist
# ax2 = fig.add_subplot(332)                          # wrn, cifar10
# ax3 = fig.add_subplot(333, sharey=ax2, sharex=ax2)  # wrn, cifar100
#
# ax4 = fig.add_subplot(334, sharey=ax2)              # lenet, mnist
# ax5 = fig.add_subplot(335, sharey=ax2, sharex=ax2)  # lenet, cifar10
# ax6 = fig.add_subplot(336, sharey=ax2, sharex=ax2)  # lenet, cifar100
#
# ax7 = fig.add_subplot(337, sharey=ax2, sharex=ax4)  # fc2net, mnist
# ax8 = fig.add_subplot(338, sharey=ax2, sharex=ax2)  # fc2net, cifar10
# ax9 = fig.add_subplot(339, sharey=ax2, sharex=ax2)  # fc2net, cifar100

""""""

def add_subplot_axes(ax, rect, facecolor='moccasin'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    # x_labelsize *= rect[2]**0.5
    # y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

fig = plt.figure(figsize=(15.0, 8.0))
subpos = [0.55, 0.3, 0.4, 0.3]
# wrn, mnist
csv_file = '/data/gilad/logs/ma_scores/wrn/mnist/log_2238_210318_ma_score_wrn_mnist_wd_0.00078-SUPERSEED=21031802/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax1 = fig.add_subplot(331)
ax1.plot(steps, values, 'r')
subax1 = add_subplot_axes(ax1, subpos)
subax1.set_ylim([99, 99.5])
subax1.set_yticks([99, 99.5])
subax1.plot(steps[-11:], values[-11:], 'r')
csv_file = '/data/gilad/logs/ma_scores/wrn/mnist/log_2238_210318_ma_score_wrn_mnist_wd_0.00078-SUPERSEED=21031802/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax1.plot(steps, values, 'black')
subax1.plot(steps[-11:], values[-11:], 'black')
ax1.set_ylim(bottom=0, top=110)
ax1.set_title('MNIST')
ax1.text(-180, 52, 'Wide ResNet 28-10', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax1.yaxis.grid()
ax1.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax1.add_patch(patches.Rectangle(xy=(400, 95), width=100, height=10, facecolor='moccasin'))

# wrn, cifar10
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax2 = fig.add_subplot(332)
ax2.plot(steps, values, 'r')
subax2 = add_subplot_axes(ax2, subpos)
subax2.set_ylim([95, 95.3])
subax2.set_yticks([95, 95.3])
subax2.plot(steps[-11:], values[-11:], 'r')
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax2.plot(steps, values, 'black')
subax2.plot(steps[-11:], values[-11:], 'black')
ax2.set_ylim(bottom=0, top=110)
ax2.set_title('CIFAR-10')
ax2.yaxis.grid()
ax2.add_patch(patches.Rectangle(xy=(40000, 90), width=10000, height=10, facecolor='moccasin'))

# wrn, cifar100
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar100/log_1444_070318_wrn_cifar100_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax3 = fig.add_subplot(333)
ax3.plot(steps, values, 'r')
subax3 = add_subplot_axes(ax3, subpos)
subax3.set_ylim([78.5, 79])
subax3.set_yticks([78.5, 79])
subax3.plot(steps[-11:], values[-11:], 'r')
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar100/log_1444_070318_wrn_cifar100_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax3.plot(steps, values, 'black')
subax3.plot(steps[-11:], values[-11:], 'black')
ax3.set_ylim(bottom=0, top=85)
ax3.set_title('CIFAR-100')
ax3.yaxis.grid()
ax3.add_patch(patches.Rectangle(xy=(40000, 75), width=10000, height=8, facecolor='moccasin'))

# lenet, mnist
csv_file = '/data/gilad/logs/ma_scores/lenet/mnist/log_2200_100318_ma_score_lenet_mnist_wd_0.0-SUPERSEED=10031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax4 = fig.add_subplot(334)
ax4.plot(steps, values, 'r')
subax4 = add_subplot_axes(ax4, subpos)
subax4.set_ylim([98.7, 99.2])
subax4.set_yticks([98.7, 99.2])
subax4.plot(steps[-11:], values[-11:], 'r')
csv_file = '/data/gilad/logs/ma_scores/lenet/mnist/log_2200_100318_ma_score_lenet_mnist_wd_0.0-SUPERSEED=10031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax4.plot(steps, values, 'black')
subax4.plot(steps[-11:], values[-11:], 'black')
ax4.set_ylim(bottom=0, top=110)
ax4.text(-180, 52, 'LeNet', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax4.yaxis.grid()
ax4.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax4.add_patch(patches.Rectangle(xy=(400, 94), width=100, height=10, facecolor='moccasin'))

# lenet, cifar10
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar10/log_2354_060318_lenet_ma_score_wd_0.008-SUPERSEED=06031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax5 = fig.add_subplot(335)
ax5.plot(steps, values, 'r')
subax5 = add_subplot_axes(ax5, subpos)
subax5.set_ylim([82, 83.6])
subax5.set_yticks([82, 83.6])
subax5.plot(steps[-11:], values[-11:], 'r')
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar10/log_2354_060318_lenet_ma_score_wd_0.008-SUPERSEED=06031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax5.plot(steps, values, 'black')
subax5.plot(steps[-11:], values[-11:], 'black')
ax5.set_ylim(bottom=0, top=110)
ax5.yaxis.grid()
ax5.add_patch(patches.Rectangle(xy=(40000, 78), width=10000, height=10, facecolor='moccasin'))


# lenet, cifar100
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar100/log_2340_090318_lenet_cifar100_wd_0.01-SUPERSEED=08031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax6 = fig.add_subplot(336)
subax6 = add_subplot_axes(ax6, subpos)
subax6.set_ylim([51.7, 54])
subax6.set_yticks([51.7, 54])
subax6.plot(steps[-11:], values[-11:], 'r')
ax6.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/lenet/cifar100/log_2340_090318_lenet_cifar100_wd_0.01-SUPERSEED=08031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
subax6.plot(steps[-11:], values[-11:], 'black')
ax6.plot(steps, values, 'black')
ax6.set_ylim(bottom=0, top=60)
ax6.yaxis.grid()
ax6.add_patch(patches.Rectangle(xy=(40000, 50), width=10000, height=6, facecolor='moccasin'))

# fc2net, mnist
csv_file = '/data/gilad/logs/ma_scores/fc2net/mnist/log_1409_140318_ma_score_fc2net_mnist_wd_0.0-SUPERSEED=14031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax7 = fig.add_subplot(337)
ax7.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/fc2net/mnist/log_1409_140318_ma_score_fc2net_mnist_wd_0.0-SUPERSEED=14031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax7.plot(steps, values, 'black')
ax7.set_ylim(bottom=0, top=110)
ax7.text(-180, 52, 'FC2Net', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax7.yaxis.grid()
ax7.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})

# fc2net, cifar10
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar10/log_1705_090318_ma_score_fc2net_cifar10_wd_0.0-SUPERSEED=08031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax8 = fig.add_subplot(338)
ax8.plot(steps, values, 'r')
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar10/log_1705_090318_ma_score_fc2net_cifar10_wd_0.0-SUPERSEED=08031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax8.plot(steps, values, 'black')
ax8.set_ylim(bottom=0, top=65)
ax8.yaxis.grid()

# fc2net, cifar100
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar100/log_1353_100318_ma_score_fc2net_cifar100_wd_0.0-SUPERSEED=10031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax9 = fig.add_subplot(339)
ax9.plot(steps, values, 'r')
subax9 = add_subplot_axes(ax9, subpos)
subax9.set_ylim([23.5, 23.7])
subax9.set_yticks([23.5, 23.7])
subax9.plot(steps[-11:], values[-11:], 'r')
csv_file = '/data/gilad/logs/ma_scores/fc2net/cifar100/log_1353_100318_ma_score_fc2net_cifar100_wd_0.0-SUPERSEED=10031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax9.plot(steps, values, 'black')
subax9.plot(steps[-11:], values[-11:], 'black')
ax9.set_ylim(bottom=0, top=30)
ax9.yaxis.grid()
ax9.add_patch(patches.Rectangle(xy=(40000, 22), width=10000, height=3, facecolor='moccasin'))
ax9.legend(['k-NN', 'DNN'], loc=(1.05, 3.1))

plt.show()
plt.savefig('knn_dnn_acuuracy_vs_iter.png', dpi=350)
