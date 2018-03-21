"""Plotting the 8 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper
import matplotlib.pyplot as plt

fig = plt.figure()
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

# wrn, cifar10
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
ax2 = fig.add_subplot(332)
ax2.plot(steps, values)
ax2.set_ylim(bottom=0, top=100)
ax2.set_title('CIFAR-10')
plt.show()