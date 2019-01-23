from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import json

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(10.0, 12.0))
plt.gcf().subplots_adjust(bottom=0.13)
layers = ['input\nimages', 'init\nconv',
          'unit_1_1', 'unit_1_2', 'unit_1_3', 'unit_1_4',
          'unit_2_1', 'unit_2_2', 'unit_2_3', 'unit_2_4',
          'unit_3_1', 'unit_3_2', 'unit_3_3', 'unit_3_4',
          'embedding\nvector']
x = np.arange(len(layers))

# wrn, cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax3 = fig.add_subplot(411)
ax3.set_xticks(x)
ax3.set_xticklabels([])
ax3.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax3.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax3.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax3.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax3.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax3.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')
ax3.tick_params(labelsize=17)

ax3.grid()
ax3.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 18})
ax3.text(-3, 60, 'CIFAR-10', va='center', rotation='vertical', fontdict={'fontsize': 18})

# wrn, random_cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax4 = fig.add_subplot(412)
ax4.set_xticks(x)
ax4.set_xticklabels([])
ax4.set_ylim(0, 104)
ax4.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax4.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax4.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax4.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax4.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax4.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')
ax4.tick_params(labelsize=17)

ax4.grid()
ax4.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 18})
ax4.text(-3, 50, 'Random CIFAR-10', va='center', rotation='vertical', fontdict={'fontsize': 18})

# wrn, cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax5 = fig.add_subplot(413)
ax5.set_xticks(x)
ax5.set_xticklabels([])
ax5.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax5.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax5.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax5.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax5.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax5.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')
ax5.tick_params(labelsize=17)

ax5.grid()
ax5.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 18})
ax5.text(-3, 50, 'CIFAR-100', va='center', rotation='vertical', fontdict={'fontsize': 18})

# wrn, random_cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax6 = fig.add_subplot(414)
ax6.set_xticks(x)
ax6.set_xticklabels(layers, fontdict={'rotation': 'vertical'})
ax6.plot(x, [v*100 for v in data['test']['layer']['knn_score']] , 'r--')
ax6.plot(x, [v*100 for v in data['test']['layer']['svm_score']] , 'b--')
ax6.plot(x, [v*100 for v in data['test']['layer']['lr_score']]  , 'g--')
ax6.plot(x, [v*100 for v in data['train']['layer']['knn_score']], 'r')
ax6.plot(x, [v*100 for v in data['train']['layer']['svm_score']], 'b')
ax6.plot(x, [v*100 for v in data['train']['layer']['lr_score']] , 'g')
ax6.tick_params(labelsize=15)

ax6.grid()
ax6.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 18})
ax6.text(-3, 50, 'Random CIFAR-100', va='center', rotation='vertical', fontdict={'fontsize': 18})


# ax4.legend(['$k$-NN (train)', 'SVM (train)', 'Logistic regression (train)', '$k$-NN (test)', 'SVM (test)', 'Logistic regression (test)'],
#            loc=(1.01, 0.0))
ax3.legend(['$k$-NN (test)', 'SVM (test)', 'LR (test)', '$k$-NN (train)', 'SVM (train)', 'LR (train)'],
           ncol=2, loc=(0.46, 0.02), fontsize=15)
# plt.tight_layout()
plt.savefig('train_test_acc_vs_layer.png')
