"""Plotting the KNN accuracy plots by wrn layers"""
from utils.plots import add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(9.0, 8.0))
plt.gcf().subplots_adjust(bottom=0.13)
# fig, axarr = plt.subplots(311, sharex=True)
subpos = [0.8, 0.25, 0.17, 0.5]
layers = ['input\nimages', 'init\nconv',
          'unit_1_1', 'unit_1_2', 'unit_1_3', 'unit_1_4',
          'unit_2_1', 'unit_2_2', 'unit_2_3', 'unit_2_4',
          'unit_3_1', 'unit_3_2', 'unit_3_3', 'unit_3_4',
          'embedding\nvector']
x = np.arange(len(layers))

# wrn, mnist
k_1   = [17.39, 54.21, 87.21, 94.71, 96.69, 97.53, 98.33, 99.06, 99.17, 99.35, 99.35, 99.29, 99.26, 99.27, 99.28]
k_5   = [20.53, 56.89, 87.94, 95.02, 96.82, 97.74, 98.4,  99.04, 99.31, 99.45, 99.47, 99.5,  99.45, 99.45, 99.44]
k_30  = [21.98, 59.04, 85.66, 93.73, 96.04, 97.02, 98.08, 98.79, 99.15, 99.36, 99.41, 99.49, 99.42, 99.46, 99.44]
k_100 = [22.71, 57.39, 82.21, 91.91, 94.79, 96.32, 97.58, 98.57, 98.98, 99.22, 99.37, 99.44, 99.46, 99.47, 99.42]
dnn   = [99.31] * len(layers)
ax1 = fig.add_subplot(311)
ax1.set_xticks(x)
ax1.set_xticklabels([])
ax1.plot(x, k_1, 'r', linestyle='-', marker='o')
ax1.plot(x, k_5, 'blue', linestyle='-', marker='o')
ax1.plot(x, k_30, 'violet', linestyle='-', marker='o')
ax1.plot(x, k_100, 'green', linestyle='-', marker='o')
ax1.plot(x, dnn, 'k--')
ax1.grid()
# ax1.plot([11.6, 13.5], [83, 95.8], 'k')
ax1.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax1.text(-2.7, 60, 'MNIST', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax1.add_patch(patches.Polygon(xy=np.array([[13.5, 102.3], [11.6, 82.7], [14.25, 82.7], [14.25, 41.33], [14.5, 95.4]]), closed=True, color='silver'))
ax1.add_patch(patches.Rectangle(xy=(13.5, 95), width=1, height=10, facecolor='moccasin'))
subax1 = add_subplot_axes(ax1, subpos)
subax1.set_ylim([99.25, 99.5])
subax1.set_xlim([13.9, 14.1])
subax1.set_xticks([14])
subax1.set_xticklabels(['embedding\nvector'])
subax1.set_yticks([99.25, 99.5])
subax1.plot(x[-1], k_1[-1], 'ro')
subax1.plot(x[-1], k_5[-1], 'bo')
subax1.plot(x[-1]+0.01, k_30[-1], 'violet', marker='o')
subax1.plot(x[-1], k_100[-1], 'green', marker='o')
subax1.plot([13,14,15], [dnn[-1]]*3, 'k--')

# wrn, cifar10
k_1   = [18.35, 34.36, 45.35, 48.06, 52.24, 54.11, 59.52, 65.37, 70.1,  73.26, 84.66, 94.53, 95.14, 95.12, 95.13]
k_5   = [20.78, 37.28, 47.8,  50.88, 55,    57.17, 62.57, 68.77, 73.89, 77.03, 86.99, 94.6,  95.17, 95.21, 95.19]
k_30  = [24.62, 39.8,  49.34, 51.99, 54.77, 57.6,  63.1,  69.26, 74.42, 77.32, 86.88, 94.48, 95.2,  95.3,  95.2]
k_100 = [25.92, 38.45, 47.18, 49.91, 52.89, 54.96, 60.74, 66.83, 72.33, 75.38, 85.83, 94.25, 95.17, 95.25, 95.18]
dnn   = [95.2] * len(layers)
ax2 = fig.add_subplot(312)
ax2.set_xticks(x)
ax2.set_xticklabels([])
ax2.set_ylim(bottom=15, top=100)
ax2.plot(x, k_1, 'r', linestyle='-', marker='o')
ax2.plot(x, k_5, 'blue', linestyle='-', marker='o')
ax2.plot(x, k_30, 'violet', linestyle='-', marker='o')
ax2.plot(x, k_100, 'green', linestyle='-', marker='o')
ax2.plot(x, dnn, 'k--')
ax2.grid()
ax2.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax2.text(-2.7, 60, 'CIFAR-10', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax2.add_patch(patches.Polygon(xy=np.array([[13.5, 98.56], [11.6, 76.75], [14.25, 76.75], [14.25, 38.25], [14.5, 90.5]]), closed=True, color='silver'))
ax2.add_patch(patches.Rectangle(xy=(13.5, 90), width=1, height=10, facecolor='moccasin'))
subax2 = add_subplot_axes(ax2, subpos)
subax2.set_ylim([95.12, 95.22])
subax2.set_xlim([13.9, 14.1])
subax2.set_xticks([14])
subax2.set_xticklabels(['embedding\nvector'])
subax2.set_yticks([95.12, 95.22])
subax2.plot(x[-1], k_1[-1], 'ro')
subax2.plot(x[-1], k_5[-1], 'bo')
subax2.plot(x[-1], k_30[-1], 'violet', marker='o')
subax2.plot(x[-1], k_100[-1], 'green', marker='o')
subax2.plot([13,14,15], [dnn[-1]]*3, 'k--')

# wrn, cifar100
k_1   = [3.79, 13.27, 21.92, 26.1,  29.59, 31.51, 34.69, 39.65, 42.85, 44.11, 51.4,  67.13, 77.36, 77.98, 78.69]
k_5   = [3.9,  12.86, 21,    25.34, 28.57, 29.73, 34.08, 39.04, 42.35, 44.39, 52.61, 69.48, 77.72, 77.82, 78.91]
k_30  = [5.37, 14.82, 22.55, 27.09, 30.45, 32.36, 35.92, 40.98, 44.37, 46.74, 54.58, 71.1,  78.01, 77.9,  78.88]
k_100 = [6.29, 14.63, 21.47, 25.31, 28.94, 30.29, 33.88, 39.02, 42.09, 44.27, 52.55, 69.79, 77.43, 77.57, 78.89]
dnn   = [78.85] * len(layers)
ax3 = fig.add_subplot(313)
ax3.set_xticks(x)
ax3.set_xticklabels(layers, fontdict={'rotation': 'vertical'})
ax3.plot(x, k_1, 'r', linestyle='-', marker='o')
ax3.plot(x, k_5, 'blue', linestyle='-', marker='o')
ax3.plot(x, k_30, 'violet', linestyle='-', marker='o')
ax3.plot(x, k_100, 'green', linestyle='-', marker='o')
ax3.plot(x, dnn, 'k--')
ax3.grid()
ax3.set_ylabel('accuracy (%)', labelpad=7, fontdict={'fontsize': 12})
ax3.text(-2.7, 42, 'CIFAR-100', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax3.add_patch(patches.Polygon(xy=np.array([[13.5, 82.4], [11.6, 56.6], [14.25, 56.6], [14.25, 18.75], [14.5, 74.5]]), closed=True, color='silver'))
ax3.add_patch(patches.Rectangle(xy=(13.5, 74), width=1, height=10, facecolor='moccasin'))
subax3 = add_subplot_axes(ax3, subpos)
subax3.set_ylim([78.66, 78.93])
subax3.set_xlim([13.9, 14.1])
subax3.set_xticks([14])
subax3.set_xticklabels(['embedding\nvector'])
subax3.set_yticks([78.66, 78.93])
subax3.plot(x[-1], k_1[-1], 'ro')
subax3.plot(x[-1], k_5[-1], 'bo')
subax3.plot(x[-1], k_30[-1], 'violet', marker='o')
subax3.plot(x[-1], k_100[-1], 'green', marker='o')
subax3.plot([13,14,15], [dnn[-1]]*3, 'k--')
ax3.legend(['K=1', 'K=5', 'K=30', 'K=100', 'DNN'], loc=(0.2, 2.2))

plt.subplots_adjust(hspace=0.05)
plt.savefig('knn_dnn_acuuracy_vs_layer_wrn.png', dpi=350)
