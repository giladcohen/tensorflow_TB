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

plt.style.use('classic')
plt.rcParams['interactive'] = True

#TODO(gilad): ADD random prints
root_dirs = ['/data/gilad/logs/metrics/wrn/mnist/log_0049_270818_metrics_w_confidence-SUPERSEED=27081800',
             '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800',
             '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800',
             '/data/gilad/logs/metrics/lenet/mnist/log_0152_140918_metrics-SUPERSEED=14091800',
             '/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800',
             '/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800',
             '/data/gilad/logs/metrics/fc2net/mnist/log_0709_150918_metrics-SUPERSEED=15091800',
             '/data/gilad/logs/metrics/fc2net/cifar10/log_1025_150918_metrics-SUPERSEED=15091800',
             '/data/gilad/logs/metrics/fc2net/cifar100/log_1025_150918_metrics-SUPERSEED=15091800']

datasets = ['mnist', 'cifar10', 'cifar100']

for dataset in datasets:
    if dataset is 'mnist':
        json_file_wrn    = os.path.join('/data/gilad/logs/metrics/wrn/mnist/log_0049_270818_metrics_w_confidence-SUPERSEED=27081800'   , 'data_for_figures', 'data.json')
        json_file_lenet  = os.path.join('/data/gilad/logs/metrics/lenet/mnist/log_0152_140918_metrics-SUPERSEED=14091800'              , 'data_for_figures', 'data.json')
        json_file_fc2net = os.path.join('/data/gilad/logs/metrics/fc2net/mnist/log_0709_150918_metrics-SUPERSEED=15091800'             , 'data_for_figures', 'data.json')
    elif dataset is 'cifar10':
        json_file_wrn    = os.path.join('/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800' , 'data_for_figures', 'data.json')
        json_file_lenet  = os.path.join('/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800'            , 'data_for_figures', 'data.json')
        json_file_fc2net = os.path.join('/data/gilad/logs/metrics/fc2net/cifar10/log_1025_150918_metrics-SUPERSEED=15091800'           , 'data_for_figures', 'data.json')
    else:
        json_file_wrn    = os.path.join('/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800', 'data_for_figures', 'data.json')
        json_file_lenet  = os.path.join('/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800'           , 'data_for_figures', 'data.json')
        json_file_fc2net = os.path.join('/data/gilad/logs/metrics/fc2net/cifar100/log_1025_150918_metrics-SUPERSEED=15091800'          , 'data_for_figures', 'data.json')


    with open(json_file_wrn) as f:
        data_wrn = json.load(f)
    with open(json_file_lenet) as f:
        data_lenet = json.load(f)
    with open(json_file_fc2net) as f:
        data_fc2net = json.load(f)

    is_randomized = 'random' in dataset

    # plotting KL divergence (average):
    # KNN || DNN
    # plt.figure()
    # plt.plot(data_wrn['test']['regular']['knn_kl_div_avg']['steps']   , data_wrn['test']['regular']['knn_kl_div_avg']['values']   , 'r')
    # plt.plot(data_lenet['test']['regular']['knn_kl_div_avg']['steps'] , data_lenet['test']['regular']['knn_kl_div_avg']['values'] , 'g')
    # plt.plot(data_fc2net['test']['regular']['knn_kl_div_avg']['steps'], data_fc2net['test']['regular']['knn_kl_div_avg']['values'], 'b')
    # plt.gca().yaxis.grid(True)
    # plt.ylabel('DNN-KNN KL divergence')
    # plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    # plt.title('Test DNN-KNN KL divergence scores on {}'.format(dataset))
    #
    # plt.figure()
    # plt.plot(data_wrn['train']['regular']['knn_kl_div_avg']['steps']   , data_wrn['train']['regular']['knn_kl_div_avg']['values']   , 'r')
    # plt.plot(data_lenet['train']['regular']['knn_kl_div_avg']['steps'] , data_lenet['train']['regular']['knn_kl_div_avg']['values'] , 'g')
    # plt.plot(data_fc2net['train']['regular']['knn_kl_div_avg']['steps'], data_fc2net['train']['regular']['knn_kl_div_avg']['values'], 'b')
    # plt.gca().yaxis.grid(True)
    # plt.ylabel('DNN-KNN KL divergence')
    # plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    # plt.title('Train DNN-KNN KL divergence scores on {}'.format(dataset))
    #
    # # SVM || DNN
    # plt.figure()
    # plt.plot(data_wrn['test']['regular']['svm_kl_div_avg']['steps']   , data_wrn['test']['regular']['svm_kl_div_avg']['values']   , 'r')
    # plt.plot(data_lenet['test']['regular']['svm_kl_div_avg']['steps'] , data_lenet['test']['regular']['svm_kl_div_avg']['values'] , 'g')
    # plt.plot(data_fc2net['test']['regular']['svm_kl_div_avg']['steps'], data_fc2net['test']['regular']['svm_kl_div_avg']['values'], 'b')
    # plt.gca().yaxis.grid(True)
    # plt.ylabel('DNN-SVM KL divergence')
    # plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    # plt.title('Test DNN-SVM KL divergence scores on {}'.format(dataset))
    #
    # plt.figure()
    # plt.plot(data_wrn['train']['regular']['svm_kl_div_avg']['steps']   , data_wrn['train']['regular']['svm_kl_div_avg']['values']   , 'r')
    # plt.plot(data_lenet['train']['regular']['svm_kl_div_avg']['steps'] , data_lenet['train']['regular']['svm_kl_div_avg']['values'] , 'g')
    # plt.plot(data_fc2net['train']['regular']['svm_kl_div_avg']['steps'], data_fc2net['train']['regular']['svm_kl_div_avg']['values'], 'b')
    # plt.gca().yaxis.grid(True)
    # plt.ylabel('DNN-SVM KL divergence')
    # plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    # plt.title('Train DNN-SVM KL divergence scores on {}'.format('mnist'))

    # LR || DNN
    # plt.figure()
    # plt.plot(data_wrn['test']['regular']['lr_kl_div_avg']['steps']   , data_wrn['test']['regular']['lr_kl_div_avg']['values']   , 'r')
    # plt.plot(data_lenet['test']['regular']['lr_kl_div_avg']['steps'] , data_lenet['test']['regular']['lr_kl_div_avg']['values'] , 'g')
    # plt.plot(data_fc2net['test']['regular']['lr_kl_div_avg']['steps'], data_fc2net['test']['regular']['lr_kl_div_avg']['values'], 'b')
    # plt.gca().yaxis.grid(True)
    # plt.ylabel('DNN-LR KL divergence score')
    # plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    # plt.title('Test DNN-SVM KL divergence scores on {}'.format(dataset))
    #
    # plt.figure()
    # plt.plot(data_wrn['train']['regular']['lr_kl_div_avg']['steps']   , data_wrn['train']['regular']['lr_kl_div_avg']['values']   , 'r')
    # plt.plot(data_lenet['train']['regular']['lr_kl_div_avg']['steps'] , data_lenet['train']['regular']['lr_kl_div_avg']['values'] , 'g')
    # plt.plot(data_fc2net['train']['regular']['lr_kl_div_avg']['steps'], data_fc2net['train']['regular']['lr_kl_div_avg']['values'], 'b')
    # plt.gca().yaxis.grid(True)
    # plt.ylabel('DNN-LR KL divergence score')
    # plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    # plt.title('Train DNN-LR KL divergence scores on {}'.format('mnist'))

    # plotting psame values
    plt.figure()
    plt.plot(data_wrn['test']['regular']['knn_psame']['steps']   , data_wrn['test']['regular']['knn_psame']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['knn_psame']['steps'] , data_lenet['test']['regular']['knn_psame']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['knn_psame']['steps'], data_fc2net['test']['regular']['knn_psame']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KNN psame score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test KNN psame scores on {}'.format(dataset))

    plt.figure()
    plt.plot(data_wrn['train']['regular']['knn_psame']['steps']   , data_wrn['train']['regular']['knn_psame']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['knn_psame']['steps'] , data_lenet['train']['regular']['knn_psame']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['knn_psame']['steps'], data_fc2net['train']['regular']['knn_psame']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KNN psame accuracy score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train KNN psame scores on {}'.format(dataset))



plt.figure(33)
plt.plot(data_wrn['test']['regular']['svm_psame']['steps'], data_wrn['test']['regular']['svm_psame']['values'], 'b')
plt.plot(data_lenet['test']['regular']['svm_psame']['steps'], data_lenet['test']['regular']['svm_psame']['values'], 'b^')
plt.plot(data_fc2net['test']['regular']['svm_psame']['steps'], data_fc2net['test']['regular']['svm_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('SVM psame accuracy score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Test SVM psame accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))

plt.figure(34)
plt.plot(data_wrn['train']['regular']['svm_psame']['steps'], data_wrn['train']['regular']['svm_psame']['values'], 'b')
plt.plot(data_lenet['train']['regular']['svm_psame']['steps'], data_lenet['train']['regular']['svm_psame']['values'], 'b^')
plt.plot(data_fc2net['train']['regular']['svm_psame']['steps'], data_fc2net['train']['regular']['svm_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('SVM psame accuracy score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Train SVM psame accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))



# for cifar10
json_file_wrn    = os.path.join('/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800', 'data_for_figures', 'data.json')
json_file_lenet  = os.path.join('/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800'           , 'data_for_figures', 'data.json')
json_file_fc2net = os.path.join('/data/gilad/logs/metrics/fc2net/cifar10/log_1025_150918_metrics-SUPERSEED=15091800'          , 'data_for_figures', 'data.json')

with open(json_file_wrn) as f:
    data_wrn = json.load(f)
with open(json_file_lenet) as f:
    data_lenet = json.load(f)
with open(json_file_fc2net) as f:
    data_fc2net = json.load(f)

is_randomized = False
plt.figure(21)
plt.plot(data_wrn['test']['regular']['knn_kl_div_avg']['steps'], data_wrn['test']['regular']['knn_kl_div_avg']['values'], 'r')
plt.plot(data_lenet['test']['regular']['knn_kl_div_avg']['steps'], data_lenet['test']['regular']['knn_kl_div_avg']['values'], 'r^')
plt.plot(data_fc2net['test']['regular']['knn_kl_div_avg']['steps'], data_fc2net['test']['regular']['knn_kl_div_avg']['values'], 'ro')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('DNN-KNN KL divergence score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.0, 0.0))
plt.title('Test DNN-KNN KL divergence scores on {} (is_randomized={})'.format('cifar10', is_randomized))

plt.figure(22)
plt.plot(data_wrn['train']['regular']['knn_kl_div_avg']['steps'], data_wrn['train']['regular']['knn_kl_div_avg']['values'], 'r')
plt.plot(data_lenet['train']['regular']['knn_kl_div_avg']['steps'], data_lenet['train']['regular']['knn_kl_div_avg']['values'], 'r^')
plt.plot(data_fc2net['train']['regular']['knn_kl_div_avg']['steps'], data_fc2net['train']['regular']['knn_kl_div_avg']['values'], 'ro')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('DNN-KNN KL divergence score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.0, 0.0))
plt.title('Train DNN-KNN KL divergence scores on {} (is_randomized={})'.format('cifar10', is_randomized))

plt.figure(23)
plt.plot(data_wrn['test']['regular']['svm_kl_div_avg']['steps'], data_wrn['test']['regular']['svm_kl_div_avg']['values'], 'r')
plt.plot(data_lenet['test']['regular']['svm_kl_div_avg']['steps'], data_lenet['test']['regular']['svm_kl_div_avg']['values'], 'r^')
plt.plot(data_fc2net['test']['regular']['svm_kl_div_avg']['steps'], data_fc2net['test']['regular']['svm_kl_div_avg']['values'], 'ro')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('DNN-SVM KL divergence score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.0, 0.0))
plt.title('Test DNN-SVM KL divergence scores on {} (is_randomized={})'.format('cifar10', is_randomized))

plt.figure(24)
plt.plot(data_wrn['train']['regular']['svm_kl_div_avg']['steps'], data_wrn['train']['regular']['svm_kl_div_avg']['values'], 'r')
plt.plot(data_lenet['train']['regular']['svm_kl_div_avg']['steps'], data_lenet['train']['regular']['svm_kl_div_avg']['values'], 'r^')
plt.plot(data_fc2net['train']['regular']['svm_kl_div_avg']['steps'], data_fc2net['train']['regular']['svm_kl_div_avg']['values'], 'ro')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('DNN-SVM KL divergence score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.0, 0.0))
plt.title('Train DNN-SVM KL divergence scores on {} (is_randomized={})'.format('cifar10', is_randomized))


# psame
plt.figure(231)
plt.plot(data_wrn['test']['regular']['knn_psame']['steps'], data_wrn['test']['regular']['knn_psame']['values'], 'b')
plt.plot(data_lenet['test']['regular']['knn_psame']['steps'], data_lenet['test']['regular']['knn_psame']['values'], 'b^')
plt.plot(data_fc2net['test']['regular']['knn_psame']['steps'], data_fc2net['test']['regular']['knn_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('KNN psame accuracy score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Test KNN psame accuracy scores on {} (is_randomized={})'.format('cifar10', is_randomized))

plt.figure(232)
plt.plot(data_wrn['train']['regular']['knn_psame']['steps'], data_wrn['train']['regular']['knn_psame']['values'], 'b')
plt.plot(data_lenet['train']['regular']['knn_psame']['steps'], data_lenet['train']['regular']['knn_psame']['values'], 'b^')
plt.plot(data_fc2net['train']['regular']['knn_psame']['steps'], data_fc2net['train']['regular']['knn_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('KNN psame accuracy score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Train KNN psame accuracy scores on {} (is_randomized={})'.format('cifar10', is_randomized))

plt.figure(233)
plt.plot(data_wrn['test']['regular']['svm_psame']['steps'], data_wrn['test']['regular']['svm_psame']['values'], 'b')
plt.plot(data_lenet['test']['regular']['svm_psame']['steps'], data_lenet['test']['regular']['svm_psame']['values'], 'b^')
plt.plot(data_fc2net['test']['regular']['svm_psame']['steps'], data_fc2net['test']['regular']['svm_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('SVM psame score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Test SVM psame accuracy scores on {} (is_randomized={})'.format('cifar10', is_randomized))

plt.figure(234)
plt.plot(data_wrn['train']['regular']['svm_psame']['steps'], data_wrn['train']['regular']['svm_psame']['values'], 'b')
plt.plot(data_lenet['train']['regular']['svm_psame']['steps'], data_lenet['train']['regular']['svm_psame']['values'], 'b^')
plt.plot(data_fc2net['train']['regular']['svm_psame']['steps'], data_fc2net['train']['regular']['svm_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('SVM psame score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Train SVM psame accuracy scores on {} (is_randomized={})'.format('cifar10', is_randomized))



# plt.figure(9)
# plt.plot(data_wrn['test']['regular']['dnn_confidence_avg']['steps'], data_wrn['test']['regular']['dnn_confidence_avg']['values'], 'r')
# plt.plot(data_lenet['test']['regular']['dnn_confidence_avg']['steps'], data_lenet['test']['regular']['dnn_confidence_avg']['values'], 'r^')
# plt.plot(data_fc2net['test']['regular']['dnn_confidence_avg']['steps'], data_fc2net['test']['regular']['dnn_confidence_avg']['values'], 'ro')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.0, 1.0)
# plt.ylabel('DNN confidence')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.0, 0.0))
# plt.title('Test DNN confidence scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(10)
# plt.plot(data_wrn['train']['regular']['dnn_confidence_avg']['steps'], data_wrn['train']['regular']['dnn_confidence_avg']['values'], 'r')
# plt.plot(data_lenet['train']['regular']['dnn_confidence_avg']['steps'], data_lenet['train']['regular']['dnn_confidence_avg']['values'], 'r^')
# plt.plot(data_fc2net['train']['regular']['dnn_confidence_avg']['steps'], data_fc2net['train']['regular']['dnn_confidence_avg']['values'], 'ro')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.0, 1.0)
# plt.ylabel('DNN confidence')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.0, 0.0))
# plt.title('Train DNN confidence scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(11)
# plt.plot(data_wrn['test']['regular']['knn_confidence_avg']['steps'], data_wrn['test']['regular']['knn_confidence_avg']['values'], 'b')
# plt.plot(data_lenet['test']['regular']['knn_confidence_avg']['steps'], data_lenet['test']['regular']['knn_confidence_avg']['values'], 'b^')
# plt.plot(data_fc2net['test']['regular']['knn_confidence_avg']['steps'], data_fc2net['test']['regular']['knn_confidence_avg']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.89, 1.0)
# plt.ylabel('KNN confidence score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(1.0, 0.0))
# plt.title('Test KNN confidence scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(12)
# plt.plot(data_wrn['train']['regular']['knn_confidence_avg']['steps'], data_wrn['train']['regular']['knn_confidence_avg']['values'], 'b')
# plt.plot(data_lenet['train']['regular']['knn_confidence_avg']['steps'], data_lenet['train']['regular']['knn_confidence_avg']['values'], 'b^')
# plt.plot(data_fc2net['train']['regular']['knn_confidence_avg']['steps'], data_fc2net['train']['regular']['knn_confidence_avg']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.89, 1.0)
# plt.ylabel('KNN confidence score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
# plt.title('Train KNN confidence scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(13)
# plt.plot(data_wrn['test']['regular']['svm_confidence_avg']['steps'], data_wrn['test']['regular']['svm_confidence_avg']['values'], 'b')
# plt.plot(data_lenet['test']['regular']['svm_confidence_avg']['steps'], data_lenet['test']['regular']['svm_confidence_avg']['values'], 'b^')
# plt.plot(data_fc2net['test']['regular']['svm_confidence_avg']['steps'], data_fc2net['test']['regular']['svm_confidence_avg']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.8, 1.0)
# plt.ylabel('SVM confidence score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(1.0, 0.0))
# plt.title('Test SVM confidence scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(14)
# plt.plot(data_wrn['train']['regular']['svm_score']['steps'], data_wrn['train']['regular']['svm_score']['values'], 'b')
# plt.plot(data_lenet['train']['regular']['svm_score']['steps'], data_lenet['train']['regular']['svm_score']['values'], 'b^')
# plt.plot(data_fc2net['train']['regular']['svm_score']['steps'], data_fc2net['train']['regular']['svm_score']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.8, 1.0)
# plt.ylabel('SVM confidence score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
# plt.title('Train SVM confidence scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(15)
# plt.plot(data_wrn['test']['regular']['lr_score']['steps'], data_wrn['test']['regular']['lr_score']['values'], 'b')
# plt.plot(data_lenet['test']['regular']['lr_score']['steps'], data_lenet['test']['regular']['lr_score']['values'], 'b^')
# plt.plot(data_fc2net['test']['regular']['lr_score']['steps'], data_fc2net['test']['regular']['lr_score']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.4, 1.0)
# plt.ylabel('Logistic Regression confidence score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
# plt.title('Test Logistic Regression confidence scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(16)
# plt.plot(data_wrn['train']['regular']['lr_confidence_avg']['steps'], data_wrn['train']['regular']['lr_confidence_avg']['values'], 'b')
# plt.plot(data_lenet['train']['regular']['lr_confidence_avg']['steps'], data_lenet['train']['regular']['lr_confidence_avg']['values'], 'b^')
# plt.plot(data_fc2net['train']['regular']['lr_confidence_avg']['steps'], data_fc2net['train']['regular']['lr_confidence_avg']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.4, 1.0)
# plt.ylabel('Logistic Regression confidence score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
# plt.title('Train Logistic Regression confidence scores on {} (is_randomized={})'.format('mnist', is_randomized))



# for cifar100
json_file_wrn    = os.path.join('/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800', 'data_for_figures', 'data.json')
json_file_lenet  = os.path.join('/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800'           , 'data_for_figures', 'data.json')
json_file_fc2net = os.path.join('/data/gilad/logs/metrics/fc2net/cifar100/log_1025_150918_metrics-SUPERSEED=15091800'          , 'data_for_figures', 'data.json')

with open(json_file_wrn) as f:
    data_wrn = json.load(f)
with open(json_file_lenet) as f:
    data_lenet = json.load(f)
with open(json_file_fc2net) as f:
    data_fc2net = json.load(f)

is_randomized = False

# psame
plt.figure(331)
plt.plot(data_wrn['test']['regular']['knn_psame']['steps'], data_wrn['test']['regular']['knn_psame']['values'], 'b')
plt.plot(data_lenet['test']['regular']['knn_psame']['steps'], data_lenet['test']['regular']['knn_psame']['values'], 'b^')
plt.plot(data_fc2net['test']['regular']['knn_psame']['steps'], data_fc2net['test']['regular']['knn_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('KNN psame accuracy score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Test KNN psame accuracy scores on {} (is_randomized={})'.format('cifar100', is_randomized))

plt.figure(332)
plt.plot(data_wrn['train']['regular']['knn_psame']['steps'], data_wrn['train']['regular']['knn_psame']['values'], 'b')
plt.plot(data_lenet['train']['regular']['knn_psame']['steps'], data_lenet['train']['regular']['knn_psame']['values'], 'b^')
plt.plot(data_fc2net['train']['regular']['knn_psame']['steps'], data_fc2net['train']['regular']['knn_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('KNN psame accuracy score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Train KNN psame accuracy scores on {} (is_randomized={})'.format('cifar100', is_randomized))

plt.figure(333)
plt.plot(data_wrn['test']['regular']['svm_psame']['steps'], data_wrn['test']['regular']['svm_psame']['values'], 'b')
plt.plot(data_lenet['test']['regular']['svm_psame']['steps'], data_lenet['test']['regular']['svm_psame']['values'], 'b^')
plt.plot(data_fc2net['test']['regular']['svm_psame']['steps'], data_fc2net['test']['regular']['svm_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('SVM psame score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Test SVM psame scores on {} (is_randomized={})'.format('cifar100', is_randomized))

plt.figure(334)
plt.plot(data_wrn['train']['regular']['svm_psame']['steps'], data_wrn['train']['regular']['svm_psame']['values'], 'b')
plt.plot(data_lenet['train']['regular']['svm_psame']['steps'], data_lenet['train']['regular']['svm_psame']['values'], 'b^')
plt.plot(data_fc2net['train']['regular']['svm_psame']['steps'], data_fc2net['train']['regular']['svm_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('SVM psame score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Train SVM psame scores on {} (is_randomized={})'.format('cifar100', is_randomized))

plt.figure(335)
plt.plot(data_wrn['test']['regular']['lr_psame']['steps'], data_wrn['test']['regular']['lr_psame']['values'], 'b')
plt.plot(data_lenet['test']['regular']['lr_psame']['steps'], data_lenet['test']['regular']['lr_psame']['values'], 'b^')
plt.plot(data_fc2net['test']['regular']['lr_psame']['steps'], data_fc2net['test']['regular']['lr_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('LR psame score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Test LR psame scores on {} (is_randomized={})'.format('cifar100', is_randomized))

plt.figure(336)
plt.plot(data_wrn['train']['regular']['lr_psame']['steps'], data_wrn['train']['regular']['lr_psame']['values'], 'b')
plt.plot(data_lenet['train']['regular']['lr_psame']['steps'], data_lenet['train']['regular']['lr_psame']['values'], 'b^')
plt.plot(data_fc2net['train']['regular']['lr_psame']['steps'], data_fc2net['train']['regular']['lr_psame']['values'], 'bo')
plt.gca().yaxis.grid(True)
plt.ylim(0.0, 1.0)
plt.ylabel('LR psame score')
plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
plt.title('Train LR psame scores on {} (is_randomized={})'.format('cifar100', is_randomized))

# accuracies
# plt.figure(1)
# plt.plot(data_wrn['test']['regular']['dnn_score']['steps'], data_wrn['test']['regular']['dnn_score']['values'], 'r')
# plt.plot(data_lenet['test']['regular']['dnn_score']['steps'], data_lenet['test']['regular']['dnn_score']['values'], 'r^')
# plt.plot(data_fc2net['test']['regular']['dnn_score']['steps'], data_fc2net['test']['regular']['dnn_score']['values'], 'ro')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.0, 1.0)
# plt.ylabel('DNN accuracy score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.0, 0.0))
# plt.title('Test DNN accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(2)
# plt.plot(data_wrn['train']['regular']['dnn_score']['steps'], data_wrn['train']['regular']['dnn_score']['values'], 'r')
# plt.plot(data_lenet['train']['regular']['dnn_score']['steps'], data_lenet['train']['regular']['dnn_score']['values'], 'r^')
# plt.plot(data_fc2net['train']['regular']['dnn_score']['steps'], data_fc2net['train']['regular']['dnn_score']['values'], 'ro')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.0, 1.0)
# plt.ylabel('DNN accuracy score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.0, 0.0))
# plt.title('Train DNN accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(3)
# plt.plot(data_wrn['test']['regular']['knn_score']['steps'], data_wrn['test']['regular']['knn_score']['values'], 'b')
# plt.plot(data_lenet['test']['regular']['knn_score']['steps'], data_lenet['test']['regular']['knn_score']['values'], 'b^')
# plt.plot(data_fc2net['test']['regular']['knn_score']['steps'], data_fc2net['test']['regular']['knn_score']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.93, 1.0)
# plt.ylabel('KNN accuracy score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(1.0, 0.0))
# plt.title('Test KNN accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(4)
# plt.plot(data_wrn['train']['regular']['knn_score']['steps'], data_wrn['train']['regular']['knn_score']['values'], 'b')
# plt.plot(data_lenet['train']['regular']['knn_score']['steps'], data_lenet['train']['regular']['knn_score']['values'], 'b^')
# plt.plot(data_fc2net['train']['regular']['knn_score']['steps'], data_fc2net['train']['regular']['knn_score']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.93, 1.0)
# plt.ylabel('KNN accuracy score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
# plt.title('Train KNN accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(5)
# plt.plot(data_wrn['test']['regular']['svm_score']['steps'], data_wrn['test']['regular']['svm_score']['values'], 'b')
# plt.plot(data_lenet['test']['regular']['svm_score']['steps'], data_lenet['test']['regular']['svm_score']['values'], 'b^')
# plt.plot(data_fc2net['test']['regular']['svm_score']['steps'], data_fc2net['test']['regular']['svm_score']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.8, 1.0)
# plt.ylabel('SVM accuracy score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(1.0, 0.0))
# plt.title('Test SVM accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(6)
# plt.plot(data_wrn['train']['regular']['svm_score']['steps'], data_wrn['train']['regular']['svm_score']['values'], 'b')
# plt.plot(data_lenet['train']['regular']['svm_score']['steps'], data_lenet['train']['regular']['svm_score']['values'], 'b^')
# plt.plot(data_fc2net['train']['regular']['svm_score']['steps'], data_fc2net['train']['regular']['svm_score']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.8, 1.0)
# plt.ylabel('SVM accuracy score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
# plt.title('Train SVM accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(7)
# plt.plot(data_wrn['test']['regular']['lr_score']['steps'], data_wrn['test']['regular']['lr_score']['values'], 'b')
# plt.plot(data_lenet['test']['regular']['lr_score']['steps'], data_lenet['test']['regular']['lr_score']['values'], 'b^')
# plt.plot(data_fc2net['test']['regular']['lr_score']['steps'], data_fc2net['test']['regular']['lr_score']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.88, 1.0)
# plt.ylabel('SVM accuracy score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
# plt.title('Test Logistic Regression accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))
#
# plt.figure(8)
# plt.plot(data_wrn['train']['regular']['lr_score']['steps'], data_wrn['train']['regular']['lr_score']['values'], 'b')
# plt.plot(data_lenet['train']['regular']['lr_score']['steps'], data_lenet['train']['regular']['lr_score']['values'], 'b^')
# plt.plot(data_fc2net['train']['regular']['lr_score']['steps'], data_fc2net['train']['regular']['lr_score']['values'], 'bo')
# plt.gca().yaxis.grid(True)
# plt.ylim(0.88, 1.0)
# plt.ylabel('SVM accuracy score')
# plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'], loc=(0.5, 0.5))
# plt.title('Train Logistic Regression accuracy scores on {} (is_randomized={})'.format('mnist', is_randomized))
