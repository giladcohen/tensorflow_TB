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
ax2.plot(data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values'], 'r')
ax2.plot(data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values'], 'b')
ax2.plot(data['test']['regular']['lr_psame']['steps'] , data['test']['regular']['lr_psame']['values'] , 'g')

ax2.set_ylim(bottom=0, top=1.05)
ax2.set_title('CIFAR-10', fontdict={'fontsize': 15})
ax2.yaxis.grid()
ax2.set_ylabel('$P_{SAME}$', labelpad=0.5, fontdict={'fontsize': 19})
ax2.get_xaxis().set_visible(False)
ax2.tick_params(labelsize=13)

# wrn, cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax3 = fig.add_subplot(322)
ax3.plot(data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values'], 'r')
ax3.plot(data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values'], 'b')
ax3.plot(data['test']['regular']['lr_psame']['steps'] , data['test']['regular']['lr_psame']['values'] , 'g')

ax3.set_ylim(bottom=0, top=1.05)
ax3.set_title('CIFAR-100', fontdict={'fontsize': 15})
ax3.yaxis.grid()
ax3.get_xaxis().set_visible(False)
ax3.tick_params(labelsize=13)

# lenet, cifar10
root_dir = '/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax5 = fig.add_subplot(323)
ax5.plot(data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values'], 'r')
ax5.plot(data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values'], 'b')
ax5.plot(data['test']['regular']['lr_psame']['steps'] , data['test']['regular']['lr_psame']['values'] , 'g')

ax5.set_ylim(bottom=0, top=1.05)
ax5.yaxis.grid()
ax5.set_ylabel('$P_{SAME}$', labelpad=0.5, fontdict={'fontsize': 19})
ax5.get_xaxis().set_visible(False)
ax5.tick_params(labelsize=13)

# lenet, cifar100
root_dir = '/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax6 = fig.add_subplot(324)
ax6.plot(data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values'], 'r')
ax6.plot(data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values'], 'b')
ax6.plot(data['test']['regular']['lr_psame']['steps'] , data['test']['regular']['lr_psame']['values'] , 'g')

ax6.set_ylim(bottom=0, top=1.05)
ax6.yaxis.grid()
ax6.get_xaxis().set_visible(False)
ax6.tick_params(labelsize=13)

# fc2net, cifar10
root_dir = '/data/gilad/logs/metrics/fc2net/cifar10/log_1025_150918_metrics-SUPERSEED=15091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax8 = fig.add_subplot(325)
ax8.plot(data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values'], 'r')
ax8.plot(data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values'], 'b')
ax8.plot(data['test']['regular']['lr_psame']['steps'] , data['test']['regular']['lr_psame']['values'] , 'g')

ax8.set_ylim(bottom=0, top=1.05)
ax8.yaxis.grid()
ax8.set_ylabel('$P_{SAME}$', labelpad=0.5, fontdict={'fontsize': 19})
ax8.tick_params(labelsize=13)

# fc2net, cifar100
root_dir = '/data/gilad/logs/metrics/fc2net/cifar100/log_1025_150918_metrics-SUPERSEED=15091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax9 = fig.add_subplot(326)
ax9.plot(data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values'], 'r')
ax9.plot(data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values'], 'b')
ax9.plot(data['test']['regular']['lr_psame']['steps'] , data['test']['regular']['lr_psame']['values'] , 'g')

ax9.set_ylim(bottom=0, top=0.7)
ax9.yaxis.grid()

ax2.legend(['$k$-NN', 'SVM', 'LR'], loc=(0.72, 0.1), fontsize=13)
ax3.legend(['$k$-NN', 'SVM', 'LR'], loc=(0.72, 0.1), fontsize=13)
ax5.legend(['$k$-NN', 'SVM', 'LR'], loc=(0.72, 0.1), fontsize=13)
ax6.legend(['$k$-NN', 'SVM', 'LR'], loc=(0.72, 0.1), fontsize=13)
ax8.legend(['$k$-NN', 'SVM', 'LR'], loc=(0.72, 0.1), fontsize=13)
ax9.legend(['$k$-NN', 'SVM', 'LR'], loc=(0.72, 0.1), fontsize=13)
ax9.tick_params(labelsize=13)


# fix subax position
# pos11 = subax1.get_position() # [[0.250367647059, 0.721470588235], [0.341544117647, 0.789411764706]]
# pos12 = [pos11.x0-0.052, pos11.y0+0.05, pos11.width, pos11.height]
# subax1.set_position(pos12)
# pos41 = subax4.get_position() # [[0.250367647059, 0.449705882353], [0.364338235294, 0.540294117647]]
# pos42 = [pos41.x0-0.052, pos41.y0, pos41.width, pos41.height]
# subax4.set_position(pos42)
plt.tight_layout()
plt.savefig('test_psame_vs_iter.png')

