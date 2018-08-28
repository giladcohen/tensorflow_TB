"""Printing all metrics, regular and by layer"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import numpy
import json

# parser = argparse.ArgumentParser(description='parser for metric printer')
# parser.add_argument('--root_dir', type=str, help='path to root dir')
# args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
import json

root_dir = '/data/gilad/logs/metrics/wrn/cifar10/log_1147_130818_metrics-SUPERSEED=13081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
plot_directory = os.path.join(root_dir, 'plots')
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)

with open(json_file) as f:
    data = json.load(f)

# regular plots
# ploting the models' scores
plt.figure(1)
steps, values = data['test']['regular']['dnn_score']['steps'], data['test']['regular']['dnn_score']['values']
plt.plot(steps, values, 'r')
steps, values = data['test']['regular']['knn_score']['steps'], data['test']['regular']['knn_score']['values']
plt.plot(steps, values, 'b')
steps, values = data['test']['regular']['svm_score']['steps'], data['test']['regular']['svm_score']['values']
plt.plot(steps, values, 'k')
steps, values = data['test']['regular']['lr_score']['steps'], data['test']['regular']['lr_score']['values']
plt.plot(steps, values, 'g')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('test score')
plt.legend(['dnn', 'knn', 'svm', 'lr'])
plt.show()

plt.figure(2)
steps, values = data['train']['regular']['dnn_score']['steps'], data['train']['regular']['dnn_score']['values']
plt.plot(steps, values, 'r')
steps, values = data['train']['regular']['knn_score']['steps'], data['train']['regular']['knn_score']['values']
plt.plot(steps, values, 'b')
steps, values = data['train']['regular']['svm_score']['steps'], data['train']['regular']['svm_score']['values']
plt.plot(steps, values, 'k')
steps, values = data['train']['regular']['lr_score']['steps'], data['train']['regular']['lr_score']['values']
plt.plot(steps, values, 'g')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('train score')
plt.legend(['dnn', 'knn', 'svm', 'lr'])
plt.show()


# ploting the psame score (correspondence to DNN)
plt.figure(3)
steps, values = data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values']
plt.plot(steps, values, 'b')
steps, values = data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values']
plt.plot(steps, values, 'k')
steps, values = data['test']['regular']['lr_psame']['steps'], data['test']['regular']['lr_psame']['values']
plt.plot(steps, values, 'g')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('test psame to DNN')
plt.legend(['knn', 'svm', 'lr'])
plt.show()

plt.figure(4)
steps, values = data['train']['regular']['knn_psame']['steps'], data['train']['regular']['knn_psame']['values']
plt.plot(steps, values, 'b')
steps, values = data['train']['regular']['svm_psame']['steps'], data['train']['regular']['svm_psame']['values']
plt.plot(steps, values, 'k')
steps, values = data['train']['regular']['lr_psame']['steps'], data['train']['regular']['lr_psame']['values']
plt.plot(steps, values, 'g')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('train psame to DNN')
plt.legend(['knn', 'svm', 'lr'])
plt.show()


# plotting confidences (avg/median) - #TODO(gilad): in the future
# plotting KL div
plt.figure(5)
steps, values = data['test']['regular']['knn_kl_div_avg']['steps'], data['test']['regular']['knn_kl_div_avg']['values']
plt.plot(steps, values, 'b')
steps, values = data['test']['regular']['lr_kl_div_avg']['steps'], data['test']['regular']['lr_kl_div_avg']['values']
plt.plot(steps, values, 'k')
steps, values = data['test']['regular']['svm_kl_div_avg']['steps'], data['test']['regular']['svm_kl_div_avg']['values']
plt.plot(steps, values, 'g')
plt.gca().yaxis.grid(True)
plt.ylabel('test KL div1 to DNN')
plt.legend(['knn', 'svm', 'lr'])
plt.show()

plt.figure(6)
steps, values = data['train']['regular']['knn_kl_div_avg']['steps'], data['train']['regular']['knn_kl_div_avg']['values']
plt.plot(steps, values, 'b')
steps, values = data['train']['regular']['lr_kl_div_avg']['steps'], data['train']['regular']['lr_kl_div_avg']['values']
plt.plot(steps, values, 'k')
steps, values = data['train']['regular']['svm_kl_div_avg']['steps'], data['train']['regular']['svm_kl_div_avg']['values']
plt.plot(steps, values, 'g')
plt.gca().yaxis.grid(True)
plt.ylabel('train KL div1 to DNN')
plt.legend(['knn', 'svm', 'lr'])
plt.show()

# plotting KL div2
plt.figure(7)
steps, values = data['test']['regular']['knn_kl_div2_avg']['steps'], data['test']['regular']['knn_kl_div2_avg']['values']
plt.plot(steps, values, 'b')
steps, values = data['test']['regular']['lr_kl_div2_avg']['steps'], data['test']['regular']['lr_kl_div2_avg']['values']
plt.plot(steps, values, 'k')
steps, values = data['test']['regular']['svm_kl_div2_avg']['steps'], data['test']['regular']['svm_kl_div2_avg']['values']
plt.plot(steps, values, 'g')
plt.gca().yaxis.grid(True)
plt.ylabel('test KL div2 to DNN')
plt.legend(['knn', 'svm', 'lr'])
plt.show()

plt.figure(8)
steps, values = data['train']['regular']['knn_kl_div2_avg']['steps'], data['train']['regular']['knn_kl_div2_avg']['values']
plt.plot(steps, values, 'b')
steps, values = data['train']['regular']['lr_kl_div2_avg']['steps'], data['train']['regular']['lr_kl_div2_avg']['values']
plt.plot(steps, values, 'k')
steps, values = data['train']['regular']['svm_kl_div2_avg']['steps'], data['train']['regular']['svm_kl_div2_avg']['values']
plt.plot(steps, values, 'g')
plt.gca().yaxis.grid(True)
plt.ylabel('train KL div2 to DNN')
plt.legend(['knn', 'svm', 'lr'])
plt.show()


# Layer printing
layers = ['input\nimages', 'init\nconv',
          'unit_1_1', 'unit_1_2', 'unit_1_3', 'unit_1_4',
          'unit_2_1', 'unit_2_2', 'unit_2_3', 'unit_2_4',
          'unit_3_1', 'unit_3_2', 'unit_3_3', 'unit_3_4',
          'embedding\nvector']
x = np.arange(len(layers))

plt.figure(9)
plt.plot(data['test']['layer']['knn_score'])







