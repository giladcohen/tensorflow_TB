"""Plotting the 9 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(15.0, 8.0))
subpos = [0.55, 0.3, 0.5, 0.4]

# wrn, mnist
root_dir = '/data/gilad/logs/metrics/wrn/mnist/log_0049_270818_metrics_w_confidence-SUPERSEED=27081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax1 = fig.add_subplot(331)
ax1.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax1.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax1.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax1.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax1 = add_subplot_axes(ax1, subpos)
# subax1.set_ylim([99.3, 99.5])
# subax1.set_yticks([99.3, 99.5])
# subax1.plot(data['test']['regular']['dnn_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-21:]], 'k')
# subax1.plot(data['test']['regular']['knn_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-21:]], 'r')
# subax1.plot(data['test']['regular']['svm_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-21:]], 'b')
# subax1.plot(data['test']['regular']['lr_score']['steps'][-21:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-21:]] , 'g')

ax1.set_ylim(bottom=0, top=110)
ax1.set_title('MNIST')
ax1.text(-300, 52, 'Wide ResNet 28-10', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax1.yaxis.grid()
ax1.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
# ax1.add_patch(patches.Polygon(xy=np.array([[800, 104], [530, 73], [970, 73], [967, 34], [1000, 94.2]]), closed=True, color='silver'))
# ax1.add_patch(patches.Rectangle(xy=(800, 95), width=200, height=10, facecolor='moccasin'))

# wrn, cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax2 = fig.add_subplot(332)
ax2.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax2.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax2.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax2.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax2 = add_subplot_axes(ax2, subpos)
# subax2.set_ylim([94.7, 95.2])
# subax2.set_yticks([94.7, 95.2])
# subax2.plot(data['test']['regular']['dnn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-11:]], 'k')
# subax2.plot(data['test']['regular']['knn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-11:]], 'r')
# subax2.plot(data['test']['regular']['svm_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-11:]], 'b')
# subax2.plot(data['test']['regular']['lr_score']['steps'][-11:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-11:]] , 'g')

ax2.set_ylim(bottom=0, top=110)
ax2.set_title('CIFAR-10')
ax2.yaxis.grid()
# ax2.add_patch(patches.Polygon(xy=np.array([[40000, 99], [27513.4, 65.6], [50000, 65.6], [50000, 32.2], [50000, 90]]), closed=True, color='silver'))
# ax2.add_patch(patches.Rectangle(xy=(40000, 90), width=10000, height=10, facecolor='moccasin'))

# wrn, cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax3 = fig.add_subplot(333)
ax3.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax3.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax3.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax3.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax3 = add_subplot_axes(ax3, subpos)
# subax3.set_ylim([78.1, 78.5])
# subax3.set_yticks([78.1, 78.5])
# subax3.plot(data['test']['regular']['dnn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-11:]], 'k')
# subax3.plot(data['test']['regular']['knn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-11:]], 'r')
# subax3.plot(data['test']['regular']['svm_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-11:]], 'b')
# subax3.plot(data['test']['regular']['lr_score']['steps'][-11:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-11:]] , 'g')

ax3.set_ylim(bottom=0, top=85)
ax3.set_title('CIFAR-100')
ax3.yaxis.grid()
# ax3.add_patch(patches.Polygon(xy=np.array([[40000, 83.1], [27700, 50.75], [50000, 50.75], [50000, 25], [50000, 75]]), closed=True, color='silver'))
# ax3.add_patch(patches.Rectangle(xy=(40000, 75), width=10000, height=8, facecolor='moccasin'))

# lenet, mnist
root_dir = '/data/gilad/logs/metrics/lenet/mnist/log_0152_140918_metrics-SUPERSEED=14091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax4 = fig.add_subplot(334)
ax4.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax4.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax4.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax4.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax4 = add_subplot_axes(ax4, subpos)
# subax4.set_ylim([98.7, 99.2])
# subax4.set_yticks([98.7, 99.2])
# subax4.plot(data['test']['regular']['dnn_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-21:]], 'k')
# subax4.plot(data['test']['regular']['knn_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-21:]], 'r')
# subax4.plot(data['test']['regular']['svm_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-21:]], 'b')
# subax4.plot(data['test']['regular']['lr_score']['steps'][-21:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-21:]] , 'g')

ax4.set_ylim(bottom=0, top=110)
ax4.text(-300, 52, 'LeNet', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax4.yaxis.grid()
ax4.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
# ax4.add_patch(patches.Polygon(xy=np.array([[800, 104], [530, 73], [970, 73], [967, 34], [1000, 94.2]]), closed=True, color='silver'))
# ax4.add_patch(patches.Rectangle(xy=(800, 95), width=200, height=10, facecolor='moccasin'))

# lenet, cifar10
root_dir = '/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax5 = fig.add_subplot(335)
ax5.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax5.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax5.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax5.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax5 = add_subplot_axes(ax5, subpos)
# subax5.set_ylim([82, 83.6])
# subax5.set_yticks([82, 83.6])
# subax5.plot(data['test']['regular']['dnn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-11:]], 'k')
# subax5.plot(data['test']['regular']['knn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-11:]], 'r')
# subax5.plot(data['test']['regular']['svm_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-11:]], 'b')
# subax5.plot(data['test']['regular']['lr_score']['steps'][-11:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-11:]] , 'g')

ax5.set_ylim(bottom=0, top=110)
ax5.yaxis.grid()
# ax5.add_patch(patches.Polygon(xy=np.array([[40000, 87.8], [27800, 66.5], [50000, 66.5], [50000, 33.1], [50000, 78.7]]), closed=True, color='silver'))
# ax5.add_patch(patches.Rectangle(xy=(40000, 78), width=10000, height=10, facecolor='moccasin'))

# lenet, cifar100
root_dir = '/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax6 = fig.add_subplot(336)
ax6.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax6.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax6.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax6.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax6 = add_subplot_axes(ax6, subpos)
# subax6.set_ylim([51.7, 54])
# subax6.set_yticks([51.7, 54])
# subax6.plot(data['test']['regular']['dnn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-11:]], 'k')
# subax6.plot(data['test']['regular']['knn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-11:]], 'r')
# subax6.plot(data['test']['regular']['svm_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-11:]], 'b')
# subax6.plot(data['test']['regular']['lr_score']['steps'][-11:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-11:]] , 'g')

ax6.set_ylim(bottom=0, top=60)
ax6.yaxis.grid()
# ax6.add_patch(patches.Polygon(xy=np.array([[40000, 56.2], [27800, 35.6], [50000, 35.6], [50000, 18], [50000, 50]]), closed=True, color='silver'))
# ax6.add_patch(patches.Rectangle(xy=(40000, 50), width=10000, height=6, facecolor='moccasin'))

# fc2net, mnist
root_dir = '/data/gilad/logs/metrics/fc2net/mnist/log_0709_150918_metrics-SUPERSEED=15091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax7 = fig.add_subplot(337)
ax7.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax7.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax7.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax7.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax7 = add_subplot_axes(ax4, subpos)
# subax7.set_ylim([98, 99.5])
# subax7.set_yticks([98, 99.5])
# subax7.plot(data['test']['regular']['dnn_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-21:]], 'r')
# subax7.plot(data['test']['regular']['knn_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-21:]], 'b')
# subax7.plot(data['test']['regular']['svm_score']['steps'][-21:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-21:]], 'k')
# subax7.plot(data['test']['regular']['lr_score']['steps'][-21:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-21:]] , 'g')

ax7.set_ylim(bottom=0, top=110)
ax7.text(-300, 52, 'MLP-640', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax7.yaxis.grid()
ax7.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
# ax7.add_patch(patches.Polygon(xy=np.array([[800, 104], [510, 73], [970, 73], [967, 40], [1000, 94.2]]), closed=True, color='silver'))
# ax7.add_patch(patches.Rectangle(xy=(800, 95), width=200, height=10, facecolor='moccasin'))

# fc2net, cifar10
root_dir = '/data/gilad/logs/metrics/fc2net/cifar10/log_1025_150918_metrics-SUPERSEED=15091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax8 = fig.add_subplot(338)
ax8.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax8.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax8.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax8.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax8 = add_subplot_axes(ax8, subpos)
# subax8.set_ylim([82, 83.6])
# subax8.set_yticks([82, 83.6])
# subax8.plot(data['test']['regular']['dnn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-11:]], 'r')
# subax8.plot(data['test']['regular']['knn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-11:]], 'b')
# subax8.plot(data['test']['regular']['svm_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-11:]], 'k')
# subax8.plot(data['test']['regular']['lr_score']['steps'][-11:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-11:]] , 'g')

ax8.set_ylim(bottom=0, top=65)
ax8.yaxis.grid()
# ax8.add_patch(patches.Polygon(xy=np.array([[40000, 87.8], [27800, 66.5], [50000, 66.5], [50000, 33.1], [50000, 78.7]]), closed=True, color='silver'))
# ax8.add_patch(patches.Rectangle(xy=(40000, 78), width=10000, height=10, facecolor='moccasin'))

# fc2net, cifar100
root_dir = '/data/gilad/logs/metrics/fc2net/cifar100/log_1025_150918_metrics-SUPERSEED=15091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax9 = fig.add_subplot(339)
ax9.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax9.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax9.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax9.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

# subax9 = add_subplot_axes(ax9, subpos)
# subax9.set_ylim([23.5, 23.7])
# subax9.set_yticks([23.5, 23.7])
# subax9.plot(data['test']['regular']['dnn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['dnn_score']['values'][-11:]], 'k')
# subax9.plot(data['test']['regular']['knn_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['knn_score']['values'][-11:]], 'r')
# subax9.plot(data['test']['regular']['svm_score']['steps'][-11:], [v * 100 for v in data['test']['regular']['svm_score']['values'][-11:]], 'b')
# subax9.plot(data['test']['regular']['lr_score']['steps'][-11:] , [v * 100 for v in data['test']['regular']['lr_score']['values'][-11:]] , 'g')

ax9.set_ylim(bottom=0, top=30)
ax9.yaxis.grid()
# ax9.add_patch(patches.Polygon(xy=np.array([[40000, 25], [27800, 18], [50000, 18], [50000, 9.1], [50000, 22.2]]), closed=True, color='silver'))
# ax9.add_patch(patches.Rectangle(xy=(40000, 22), width=10000, height=3, facecolor='moccasin'))

ax1.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))
ax2.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))
ax3.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))
ax4.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))
ax5.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))
ax6.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))
ax7.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))
ax8.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))
ax9.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.6, 0.1))


# fix subax position
# pos11 = subax1.get_position() # [[0.250367647059, 0.721470588235], [0.341544117647, 0.789411764706]]
# pos12 = [pos11.x0-0.052, pos11.y0+0.05, pos11.width, pos11.height]
# subax1.set_position(pos12)
# pos41 = subax4.get_position() # [[0.250367647059, 0.449705882353], [0.364338235294, 0.540294117647]]
# pos42 = [pos41.x0-0.052, pos41.y0, pos41.width, pos41.height]
# subax4.set_position(pos42)
plt.savefig('test_acc_vs_iter.png')

