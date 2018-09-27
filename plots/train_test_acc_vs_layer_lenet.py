from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import json

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(9.0, 8.0))
plt.gcf().subplots_adjust(bottom=0.13)
layers = ['input\nimages',
          'conv1', 'pool1', 'conv2', 'pool2',
          'embedding\nvector']
x = np.arange(len(layers))

# lenet, cifar10
root_dir = '/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax1 = fig.add_subplot(411)
ax1.set_xticks(x)
ax1.set_xticklabels([])
ax1.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax1.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax1.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax1.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax1.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax1.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')

ax1.grid()
ax1.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax1.text(-0.95, 60, 'CIFAR-10', va='center', rotation='vertical', fontdict={'fontsize': 13})

# lenet, random_cifar10
root_dir = '/data/gilad/logs/metrics/lenet/cifar10/random/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax2 = fig.add_subplot(412)
ax2.set_xticks(x)
ax2.set_xticklabels([])
ax2.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax2.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax2.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax2.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax2.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax2.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')

ax2.grid()
ax2.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax2.text(-0.95, 60, 'Random CIFAR-10', va='center', rotation='vertical', fontdict={'fontsize': 13})

# lenet, cifar100
root_dir = '/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax3 = fig.add_subplot(413)
ax3.set_xticks(x)
ax3.set_xticklabels([])
ax3.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax3.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax3.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax3.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax3.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax3.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')

ax3.grid()
ax3.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax3.text(-0.95, 50, 'CIFAR-100', va='center', rotation='vertical', fontdict={'fontsize': 13})

# lenet, random_cifar100
root_dir = '/data/gilad/logs/metrics/lenet/cifar100/random/log_1319_120918_metrics-SUPERSEED=12091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax4 = fig.add_subplot(414)
ax4.set_xticks(x)
ax4.set_xticklabels(layers, fontdict={'rotation': 'vertical'})
ax4.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax4.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax4.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax4.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax4.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax4.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')

ax4.grid()
ax4.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax4.text(-0.95, 30, 'Random CIFAR-100', va='center', rotation='vertical', fontdict={'fontsize': 13})
# ax4.legend(['$k$-NN (train)', 'SVM (train)', 'Logistic regression (train)', '$k$-NN (test)', 'SVM (test)', 'Logistic regression (test)'],
#            loc=(1.01, 0.0))
ax4.legend(['$k$-NN (train)', 'SVM (train)', 'Logistic regression (train)', '$k$-NN (test)', 'SVM (test)', 'Logistic regression (test)'],
           loc=(0.01, 0.1))

plt.savefig('train_test_acc_vs_layer_lenet.png')
