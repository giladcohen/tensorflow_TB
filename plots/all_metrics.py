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
steps, dnn_score = data['test']['regular']['dnn_score']['steps'], data['test']['regular']['dnn_score']['values']
plt.plot(steps, dnn_score, 'r')
steps, knn_score = data['test']['regular']['knn_score']['steps'], data['test']['regular']['knn_score']['values']
plt.plot(steps, knn_score, 'b')
steps, svm_score = data['test']['regular']['svm_score']['steps'], data['test']['regular']['svm_score']['values']
plt.plot(steps, svm_score, 'k')
steps, lr_score = data['test']['regular']['lr_score']['steps'], data['test']['regular']['lr_score']['values']
plt.plot(steps, lr_score, 'g')
plt.gca().yaxis.grid(True)

plt.figure(2)
steps, dnn_score = data['train']['regular']['dnn_score']['steps'], data['train']['regular']['dnn_score']['values']
plt.plot(steps, dnn_score, 'r')
steps, knn_score = data['train']['regular']['knn_score']['steps'], data['train']['regular']['knn_score']['values']
plt.plot(steps, knn_score, 'b')
steps, svm_score = data['train']['regular']['svm_score']['steps'], data['train']['regular']['svm_score']['values']
plt.plot(steps, svm_score, 'k')
steps, lr_score = data['train']['regular']['lr_score']['steps'], data['train']['regular']['lr_score']['values']
plt.plot(steps, lr_score, 'g')
plt.gca().yaxis.grid(True)



plt.show()










