"""Plotting the 9 KNN accuracy plots"""
from tensorflow_TB.utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(15.0, 8.0))
subpos = [0.55, 0.3, 0.5, 0.4]

# wrn, cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax2 = fig.add_subplot(111)
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

ax2.set_ylim(bottom=0, top=100)
ax2.set_title('CIFAR-10', fontdict={'fontsize': 20})
ax2.yaxis.grid()
ax2.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 20})
ax2.legend(['DNN', '$k$-NN', 'SVM', 'LR'], loc=(0.8, 0.62), fontsize=20)
ax2.tick_params(labelsize=20)
ax2.set_xlabel('iteration step', labelpad=0.5, fontdict={'fontsize': 20})

# ax2.add_patch(patches.Polygon(xy=np.array([[40000, 99], [27513.4, 65.6], [50000, 65.6], [50000, 32.2], [50000, 90]]), closed=True, color='silver'))
# ax2.add_patch(patches.Rectangle(xy=(40000, 90), width=10000, height=10, facecolor='moccasin'))

plt.tight_layout()
plt.savefig('test_acc_vs_iter_just_wrn_cifar10.png')

