from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import json

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(9.0, 12.0))
plt.gcf().subplots_adjust(bottom=0.13)
layers = ['input\nimages', 'init\nconv',
          'unit_1_1', 'unit_1_2', 'unit_1_3', 'unit_1_4',
          'unit_2_1', 'unit_2_2', 'unit_2_3', 'unit_2_4',
          'unit_3_1', 'unit_3_2', 'unit_3_3', 'unit_3_4',
          'embedding\nvector']
x = np.arange(len(layers))

# wrn, mnist
root_dir = '/data/gilad/logs/metrics/wrn/mnist/log_0049_270818_metrics_w_confidence-SUPERSEED=27081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax1 = fig.add_subplot(611)
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
ax1.text(-2.7, 60, 'MNIST', va='center', rotation='vertical', fontdict={'fontsize': 13})

# wrn, random_mnist
root_dir = '/data/gilad/logs/metrics/wrn/mnist/random/log_0420_280918_metrics-SUPERSEED=28091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax2 = fig.add_subplot(612)
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
ax2.text(-2.7, 50, 'Random MNIST', va='center', rotation='vertical', fontdict={'fontsize': 13})

# wrn, cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax3 = fig.add_subplot(613)
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
ax3.text(-2.7, 60, 'CIFAR-10', va='center', rotation='vertical', fontdict={'fontsize': 13})

# wrn, random_cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax4 = fig.add_subplot(614)
ax4.set_xticks(x)
ax4.set_xticklabels([])
ax4.set_ylim(0, 104)
ax4.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax4.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax4.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax4.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax4.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax4.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')

ax4.grid()
ax4.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax4.text(-2.7, 60, 'Random CIFAR-10', va='center', rotation='vertical', fontdict={'fontsize': 13})

# wrn, cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax5 = fig.add_subplot(615)
ax5.set_xticks(x)
ax5.set_xticklabels([])
ax5.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax5.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax5.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax5.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax5.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax5.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')

ax5.grid()
ax5.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax5.text(-2.7, 60, 'CIFAR-100', va='center', rotation='vertical', fontdict={'fontsize': 13})

# wrn, random_cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax6 = fig.add_subplot(616)
ax6.set_xticks(x)
ax6.set_xticklabels(layers, fontdict={'rotation': 'vertical'})
ax6.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax6.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax6.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax6.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax6.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax6.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')

ax6.grid()
ax6.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})
ax6.text(-2.7, 50, 'Random CIFAR-100', va='center', rotation='vertical', fontdict={'fontsize': 13})


# ax4.legend(['$k$-NN (train)', 'SVM (train)', 'Logistic regression (train)', '$k$-NN (test)', 'SVM (test)', 'Logistic regression (test)'],
#            loc=(1.01, 0.0))
ax1.legend(['$k$-NN (test)', 'SVM (test)', 'LR (test)', '$k$-NN (train)', 'SVM (train)', 'LR (train)'],
           ncol=2, loc=(0.6, 0.35))

plt.savefig('train_test_acc_vs_layer.png')
