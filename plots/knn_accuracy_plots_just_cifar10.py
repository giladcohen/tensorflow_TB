"""Plotting the 8 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(10, 8.0))
subpos = [0.55, 0.3, 0.4, 0.3]


# wrn, cifar10
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___knn_score'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = [steps[0]]  + steps[10:]
values = [values[0]] + values[10:]
ax2 = fig.add_subplot(111)
ax2.plot(steps, values, 'r')
subax2 = add_subplot_axes(ax2, subpos)
subax2.set_xticks([40000, 42000, 44000, 46000, 48000, 50000])
subax2.set_xticklabels(['40', '42', '44', '46', '48', '50'], fontdict={'fontsize': 13})
subax2.set_ylim([95, 95.3])
subax2.set_yticks([95, 95.3])
subax2.plot(steps[-11:], values[-11:], 'r')
for item in subax2.get_yticklabels():
    item.set_fontsize(13)
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___score'
steps, values = load_data_from_csv_wrapper(csv_file)
steps  = [steps[0]]  + steps[10:]
values = [values[0]] + values[10:]
ax2.plot(steps, values, 'black')
subax2.plot(steps[-11:], values[-11:], 'black')
ax2.set_ylim(bottom=0, top=110)
ax2.set_title('CIFAR-10')
ax2.yaxis.grid()
ax2.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
ax2.set_xticklabels(['0', '10', '20', '30', '40', '50'], fontdict={'fontsize': 13})
ax2.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax2.set_ylabel('test accuracy (%)', fontdict={'fontsize': 14})
for item in ax2.get_yticklabels():
    item.set_fontsize(13)
ax2.add_patch(patches.Polygon(xy=np.array([[40000, 99], [27513.4, 65.6], [49732, 65.6], [49732, 32.2], [50000, 90]]), closed=True, color='silver'))
ax2.add_patch(patches.Rectangle(xy=(40000, 90), width=10000, height=10, facecolor='moccasin'))

ax2.legend(['k-NN', 'DNN'], loc=(0.05, 0.86), prop={'size': 16})
# plt.show()
plt.savefig('knn_dnn_accuracy_just_cifar10_vs_iter.png', dpi=350)
