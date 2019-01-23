"""Plotting the 9 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(10.0, 8.0))
subpos = [0.55, 0.3, 0.5, 0.4]

# wrn, cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax2 = fig.add_subplot(321)
ax2.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax2.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax2.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax2.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

ax2.set_ylim(bottom=0, top=110)
ax2.set_title('CIFAR-10', fontdict={'fontsize': 15})
# ax2.text(-15000, 52, 'Wide ResNet 28-10', va='center', rotation='vertical', fontdict={'fontsize': 15})
ax2.yaxis.grid()
ax2.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 15})
ax2.get_xaxis().set_visible(False)
ax2.tick_params(labelsize=13)

# ax2.add_patch(patches.Polygon(xy=np.array([[40000, 99], [27513.4, 65.6], [50000, 65.6], [50000, 32.2], [50000, 90]]), closed=True, color='silver'))
# ax2.add_patch(patches.Rectangle(xy=(40000, 90), width=10000, height=10, facecolor='moccasin'))

# wrn, cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax3 = fig.add_subplot(322)
ax3.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax3.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax3.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax3.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

ax3.set_ylim(bottom=0, top=85)
ax3.set_title('CIFAR-100', fontdict={'fontsize': 15})
ax3.yaxis.grid()
ax3.get_xaxis().set_visible(False)
ax3.tick_params(labelsize=13)
# ax3.add_patch(patches.Polygon(xy=np.array([[40000, 83.1], [27700, 50.75], [50000, 50.75], [50000, 25], [50000, 75]]), closed=True, color='silver'))
# ax3.add_patch(patches.Rectangle(xy=(40000, 75), width=10000, height=8, facecolor='moccasin'))

# lenet, cifar10
root_dir = '/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax5 = fig.add_subplot(323)
ax5.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax5.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax5.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax5.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

ax5.set_ylim(bottom=0, top=110)
# ax5.text(-12000, 52, 'LeNet', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax5.yaxis.grid()
ax5.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 15})
ax5.get_xaxis().set_visible(False)
ax5.tick_params(labelsize=13)
# ax5.add_patch(patches.Polygon(xy=np.array([[40000, 87.8], [27800, 66.5], [50000, 66.5], [50000, 33.1], [50000, 78.7]]), closed=True, color='silver'))
# ax5.add_patch(patches.Rectangle(xy=(40000, 78), width=10000, height=10, facecolor='moccasin'))

# lenet, cifar100
root_dir = '/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax6 = fig.add_subplot(324)
ax6.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax6.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax6.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax6.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

ax6.set_ylim(bottom=0, top=60)
ax6.yaxis.grid()
ax6.get_xaxis().set_visible(False)
ax6.tick_params(labelsize=13)
# ax6.add_patch(patches.Polygon(xy=np.array([[40000, 56.2], [27800, 35.6], [50000, 35.6], [50000, 18], [50000, 50]]), closed=True, color='silver'))
# ax6.add_patch(patches.Rectangle(xy=(40000, 50), width=10000, height=6, facecolor='moccasin'))

# fc2net, cifar10
root_dir = '/data/gilad/logs/metrics/fc2net/cifar10/log_1025_150918_metrics-SUPERSEED=15091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax8 = fig.add_subplot(325)
ax8.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax8.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax8.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax8.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

ax8.set_ylim(bottom=0, top=65)
# ax8.text(-12000, 30, 'MLP-640', va='center', rotation='vertical', fontdict={'fontsize': 13})
ax8.set_ylabel('accuracy (%)', labelpad=6, fontdict={'fontsize': 15})
ax8.yaxis.grid()
ax8.tick_params(labelsize=13)

# ax8.add_patch(patches.Polygon(xy=np.array([[40000, 87.8], [27800, 66.5], [50000, 66.5], [50000, 33.1], [50000, 78.7]]), closed=True, color='silver'))
# ax8.add_patch(patches.Rectangle(xy=(40000, 78), width=10000, height=10, facecolor='moccasin'))

# fc2net, cifar100
root_dir = '/data/gilad/logs/metrics/fc2net/cifar100/log_1025_150918_metrics-SUPERSEED=15091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax9 = fig.add_subplot(326)
ax9.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax9.plot(data['test']['regular']['knn_score']['steps'], [v * 100 for v in data['test']['regular']['knn_score']['values']], 'r')
ax9.plot(data['test']['regular']['svm_score']['steps'], [v * 100 for v in data['test']['regular']['svm_score']['values']], 'b')
ax9.plot(data['test']['regular']['lr_score']['steps'] , [v * 100 for v in data['test']['regular']['lr_score']['values']] , 'g')

ax9.set_ylim(bottom=0, top=30)
ax9.yaxis.grid()
ax9.tick_params(labelsize=13)
# ax9.add_patch(patches.Polygon(xy=np.array([[40000, 25], [27800, 18], [50000, 18], [50000, 9.1], [50000, 22.2]]), closed=True, color='silver'))
# ax9.add_patch(patches.Rectangle(xy=(40000, 22), width=10000, height=3, facecolor='moccasin'))

ax2.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.65, 0.1), fontsize=15)
ax3.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.65, 0.1), fontsize=15)
ax5.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.65, 0.1), fontsize=15)
ax6.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.65, 0.1), fontsize=15)
ax8.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.65, 0.1), fontsize=15)
ax9.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.65, 0.1), fontsize=15)


# fix subax position
# pos11 = subax1.get_position() # [[0.250367647059, 0.721470588235], [0.341544117647, 0.789411764706]]
# pos12 = [pos11.x0-0.052, pos11.y0+0.05, pos11.width, pos11.height]
# subax1.set_position(pos12)
# pos41 = subax4.get_position() # [[0.250367647059, 0.449705882353], [0.364338235294, 0.540294117647]]
# pos42 = [pos41.x0-0.052, pos41.y0, pos41.width, pos41.height]
# subax4.set_position(pos42)
plt.tight_layout()
plt.savefig('test_acc_vs_iter.png', dpi=350)

