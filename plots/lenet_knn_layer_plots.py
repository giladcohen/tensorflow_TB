"""Plotting the KNN accuracy plots by lenet layers"""
from utils.plots import add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(9.0, 8.0))
plt.gcf().subplots_adjust(bottom=0.13)
subpos = [0.8, 0.25, 0.17, 0.5]
layers = ['input\nimages',
          'conv1', 'pool1', 'conv2', 'pool2',
          'embedding\nvector']
x = np.arange(len(layers))

# lenet, mnist
k_1   = [11.48, 72.96, 66.98, 97.66, 97.62, 99.27]
k_5   = [12.83, 75.25, 69.8,  97.76, 97.52, 99.22]
k_30  = [13.86, 73.37, 70.18, 97,    96.88, 99.06]
k_100 = [15.14, 69.76, 67.68, 96.06, 95.84, 98.94]
dnn   = [98.84] * len(layers)
ax1 = fig.add_subplot(311)
ax1.set_xticks(x)
ax1.set_xticklabels([])
ax1.plot(x, k_1, 'r', linestyle='-', marker='o')
ax1.plot(x, k_5, 'blue', linestyle='-', marker='o')
ax1.plot(x, k_30, 'violet', linestyle='-', marker='o')
ax1.plot(x, k_100, 'green', linestyle='-', marker='o')
ax1.plot(x, dnn, 'k--')
ax1.set_ylim(bottom=10, top=105)
ax1.grid()
ax1.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax1.text(-0.95, 60, 'MNIST', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax1.add_patch(patches.Polygon(xy=np.array([[4.75, 102.3], [4.17, 83.4], [5.09, 83.4], [5.09, 39.2], [5.25, 92.7]]), closed=True, color='silver'))
ax1.add_patch(patches.Rectangle(xy=(4.75, 93), width=0.5, height=10, facecolor='moccasin'))
subax1 = add_subplot_axes(ax1, subpos)
subax1.set_ylim([98.8, 99.3])
subax1.set_xlim([4.9, 5.1])
subax1.set_xticks([5])
subax1.set_xticklabels(['embedding\nvector'])
subax1.set_yticks([98.8, 99.3])
subax1.plot(x[-1], k_1[-1], 'ro')
subax1.plot(x[-1], k_5[-1], 'bo')
subax1.plot(x[-1], k_30[-1], 'violet', marker='o')
subax1.plot(x[-1], k_100[-1], 'green', marker='o')
subax1.plot([4, 5, 6], [dnn[-1]]*3, 'k--')
#
# lenet, cifar10
k_1   = [16.85, 42.74, 42.93, 58.98, 58.69, 83.09]
k_5   = [18.8,  45.27, 45.37, 62.78, 62.53, 83.38]
k_30  = [22.57, 47.7,  48.19, 63.73, 63.4,  83.49]
k_100 = [24.42, 46.45, 46.77, 61.92, 62.05, 83.28]
dnn   = [84.75] * len(layers)
ax2 = fig.add_subplot(312)
ax2.set_xticks(x)
ax2.set_xticklabels([])
ax2.plot(x, k_1, 'r', linestyle='-', marker='o')
ax2.plot(x, k_5, 'blue', linestyle='-', marker='o')
ax2.plot(x, k_30, 'violet', linestyle='-', marker='o')
ax2.plot(x, k_100, 'green', linestyle='-', marker='o')
ax2.plot(x, dnn, 'k--')
ax2.grid()
ax2.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax2.text(-0.95, 50, 'CIFAR-10', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax2.add_patch(patches.Polygon(xy=np.array([[4.75, 87.4], [4.17, 67.8], [5.09, 67.8], [5.09, 34.2], [5.25, 77.4]]), closed=True, color='silver'))
ax2.add_patch(patches.Rectangle(xy=(4.75, 78), width=0.5, height=10, facecolor='moccasin'))
subax2 = add_subplot_axes(ax2, subpos)
subax2.set_ylim([83.0, 85.0])
subax2.set_xlim([4.9, 5.1])
subax2.set_xticks([5])
subax2.set_xticklabels(['embedding\nvector'])
subax2.set_yticks([83.0, 85.0])
subax2.plot(x[-1], k_1[-1], 'ro')
subax2.plot(x[-1]+0.015, k_5[-1], 'bo')
subax2.plot(x[-1], k_30[-1], 'violet', marker='o')
subax2.plot(x[-1], k_100[-1], 'green', marker='o')
subax2.plot([4, 5, 6], [dnn[-1]]*3, 'k--')

# lenet, cifar10
k_1   = [2.9,  20.26, 20.31, 32.11, 32.09, 51.87]
k_5   = [3.29, 18.98, 19.28, 31.9,  32.27, 52.44]
k_30  = [4.63, 20.9,  21.51, 33.96, 34.1,  52.2]
k_100 = [5.54, 19.98, 20.61, 31.86, 32.14, 52.71]
dnn   = [53.35] * len(layers)
ax3 = fig.add_subplot(313)
ax3.set_xticks(x)
ax3.set_xticklabels(layers, fontdict={'rotation': 'vertical'})
ax3.plot(x, k_1, 'r', linestyle='-', marker='o')
ax3.plot(x, k_5, 'blue', linestyle='-', marker='o')
ax3.plot(x, k_30, 'violet', linestyle='-', marker='o')
ax3.plot(x, k_100, 'green', linestyle='-', marker='o')
ax3.plot(x, dnn, 'k--')
ax3.grid()
ax3.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax3.text(-0.95, 28, 'CIFAR-100', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax3.add_patch(patches.Polygon(xy=np.array([[4.75, 55.1], [4.17, 38.4], [5.09, 38.4], [5.09, 12.7], [5.25, 49.5]]), closed=True, color='silver'))
ax3.add_patch(patches.Rectangle(xy=(4.75, 50), width=0.5, height=5, facecolor='moccasin'))
subax3 = add_subplot_axes(ax3, subpos)
subax3.set_ylim([51.5, 53.6])
subax3.set_xlim([4.9, 5.1])
subax3.set_xticks([5])
subax3.set_xticklabels(['embedding\nvector'])
subax3.set_yticks([51.5, 53.6])
subax3.plot(x[-1], k_1[-1], 'ro')
subax3.plot(x[-1], k_5[-1], 'bo')
subax3.plot(x[-1], k_30[-1], 'violet', marker='o')
subax3.plot(x[-1], k_100[-1], 'green', marker='o')
subax3.plot([4, 5, 6], [dnn[-1]]*3, 'k--')
ax3.legend(['K=1', 'K=5', 'K=30', 'K=100', 'DNN'], loc=(0.24, 2.12))

plt.subplots_adjust(hspace=0.05)
plt.savefig('knn_dnn_acuuracy_vs_layer_lenet.png', dpi=350)

