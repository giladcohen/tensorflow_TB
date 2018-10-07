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

datasets = ['mnist', 'random_mnist', 'cifar10', 'random_cifar10', 'cifar100', 'random_cifar100']
plot_root = '/data/gilad/logs/metrics/all_plots/Arch_Comparison'

for dataset in datasets:
    plot_directory = os.path.join(plot_root, dataset)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    if dataset is 'mnist':
        json_file_wrn    = '/data/gilad/logs/metrics/wrn/mnist/log_0049_270818_metrics_w_confidence-SUPERSEED=27081800'
        json_file_lenet  = '/data/gilad/logs/metrics/lenet/mnist/log_0152_140918_metrics-SUPERSEED=14091800'
        json_file_fc2net = '/data/gilad/logs/metrics/fc2net/mnist/log_0709_150918_metrics-SUPERSEED=15091800'
    elif dataset is 'random_mnist':
        json_file_wrn    = '/data/gilad/logs/metrics/wrn/mnist/random/log_0420_280918_metrics-SUPERSEED=28091800'
        json_file_lenet  = '/data/gilad/logs/metrics/lenet/mnist/random/log_0420_280918_metrics-SUPERSEED=28091800'
        json_file_fc2net = '/data/gilad/logs/metrics/fc2net/mnist/random/log_0420_280918_metrics-SUPERSEED=28091800'
    elif dataset is 'cifar10':
        json_file_wrn    = '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
        json_file_lenet  = '/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800'
        json_file_fc2net = '/data/gilad/logs/metrics/fc2net/cifar10/log_1025_150918_metrics-SUPERSEED=15091800'
    elif dataset is 'random_cifar10':
        json_file_wrn    = '/data/gilad/logs/metrics/wrn/cifar10/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
        json_file_lenet  = '/data/gilad/logs/metrics/lenet/cifar10/random/log_1319_120918_metrics-SUPERSEED=12091800'
        json_file_fc2net = '/data/gilad/logs/metrics/fc2net/cifar10/random/log_1025_150918_metrics-SUPERSEED=15091800'
    elif dataset is 'cifar100':
        json_file_wrn    = '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
        json_file_lenet  = '/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800'
        json_file_fc2net = '/data/gilad/logs/metrics/fc2net/cifar100/log_1025_150918_metrics-SUPERSEED=15091800'
    elif dataset is 'random_cifar100':
        json_file_wrn    = '/data/gilad/logs/metrics/wrn/cifar100/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
        json_file_lenet  = '/data/gilad/logs/metrics/lenet/cifar100/random/log_1319_120918_metrics-SUPERSEED=12091800'
        json_file_fc2net = '/data/gilad/logs/metrics/fc2net/cifar100/random/log_1025_150918_metrics-SUPERSEED=15091800'
    else:
        err_str = 'dataset {} was not found'.format(dataset)
        raise AssertionError(err_str)

    json_file_wrn    = os.path.join(json_file_wrn   , 'data_for_figures', 'data.json')
    json_file_lenet  = os.path.join(json_file_lenet , 'data_for_figures', 'data.json')
    json_file_fc2net = os.path.join(json_file_fc2net, 'data_for_figures', 'data.json')
    with open(json_file_wrn) as f:
        data_wrn = json.load(f)
    with open(json_file_lenet) as f:
        data_lenet = json.load(f)
    with open(json_file_fc2net) as f:
        data_fc2net = json.load(f)

    # plotting KL divergence (average):
    # KNN || DNN
    plt.figure()
    plt.plot(data_wrn['test']['regular']['knn_kl_div_avg']['steps']   , data_wrn['test']['regular']['knn_kl_div_avg']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['knn_kl_div_avg']['steps'] , data_lenet['test']['regular']['knn_kl_div_avg']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['knn_kl_div_avg']['steps'], data_fc2net['test']['regular']['knn_kl_div_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('DNN-KNN KL divergence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test DNN-KNN KL divergence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_dnn_knn_KL_divergence'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['knn_kl_div_avg']['steps']   , data_wrn['train']['regular']['knn_kl_div_avg']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['knn_kl_div_avg']['steps'] , data_lenet['train']['regular']['knn_kl_div_avg']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['knn_kl_div_avg']['steps'], data_fc2net['train']['regular']['knn_kl_div_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('DNN-KNN KL divergence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train DNN-KNN KL divergence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_dnn_knn_KL_divergence'))
    plt.close()

    # SVM || DNN
    plt.figure()
    plt.plot(data_wrn['test']['regular']['svm_kl_div_avg']['steps']   , data_wrn['test']['regular']['svm_kl_div_avg']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['svm_kl_div_avg']['steps'] , data_lenet['test']['regular']['svm_kl_div_avg']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['svm_kl_div_avg']['steps'], data_fc2net['test']['regular']['svm_kl_div_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('DNN-SVM KL divergence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test DNN-SVM KL divergence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_dnn_svm_KL_divergence'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['svm_kl_div_avg']['steps']   , data_wrn['train']['regular']['svm_kl_div_avg']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['svm_kl_div_avg']['steps'] , data_lenet['train']['regular']['svm_kl_div_avg']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['svm_kl_div_avg']['steps'], data_fc2net['train']['regular']['svm_kl_div_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('DNN-SVM KL divergence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train DNN-SVM KL divergence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_dnn_svm_KL_divergence'))
    plt.close()


    # LR || DNN
    plt.figure()
    plt.plot(data_wrn['test']['regular']['lr_kl_div_avg']['steps']   , data_wrn['test']['regular']['lr_kl_div_avg']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['lr_kl_div_avg']['steps'] , data_lenet['test']['regular']['lr_kl_div_avg']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['lr_kl_div_avg']['steps'], data_fc2net['test']['regular']['lr_kl_div_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('DNN-LR KL divergence score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test DNN-LR KL divergence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_dnn_lr_KL_divergence'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['lr_kl_div_avg']['steps']   , data_wrn['train']['regular']['lr_kl_div_avg']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['lr_kl_div_avg']['steps'] , data_lenet['train']['regular']['lr_kl_div_avg']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['lr_kl_div_avg']['steps'], data_fc2net['train']['regular']['lr_kl_div_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('DNN-LR KL divergence score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train DNN-LR KL divergence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_dnn_lr_KL_divergence'))
    plt.close()

    # plotting psame values
    # KNN
    plt.figure()
    plt.plot(data_wrn['test']['regular']['knn_psame']['steps']   , data_wrn['test']['regular']['knn_psame']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['knn_psame']['steps'] , data_lenet['test']['regular']['knn_psame']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['knn_psame']['steps'], data_fc2net['test']['regular']['knn_psame']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KNN psame score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test KNN psame scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_dnn_knn_psame'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['knn_psame']['steps']   , data_wrn['train']['regular']['knn_psame']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['knn_psame']['steps'] , data_lenet['train']['regular']['knn_psame']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['knn_psame']['steps'], data_fc2net['train']['regular']['knn_psame']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KNN psame accuracy score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train KNN psame scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_dnn_knn_psame'))
    plt.close()

    # SVM
    plt.figure()
    plt.plot(data_wrn['test']['regular']['svm_psame']['steps']   , data_wrn['test']['regular']['svm_psame']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['svm_psame']['steps'] , data_lenet['test']['regular']['svm_psame']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['svm_psame']['steps'], data_fc2net['test']['regular']['svm_psame']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('SVM psame score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test SVM psame scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_dnn_svm_psame'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['svm_psame']['steps']   , data_wrn['train']['regular']['svm_psame']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['svm_psame']['steps'] , data_lenet['train']['regular']['svm_psame']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['svm_psame']['steps'], data_fc2net['train']['regular']['svm_psame']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('SVM psame score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train SVM psame scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_dnn_svm_psame'))
    plt.close()

    # LR
    plt.figure()
    plt.plot(data_wrn['test']['regular']['lr_psame']['steps']   , data_wrn['test']['regular']['lr_psame']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['lr_psame']['steps'] , data_lenet['test']['regular']['lr_psame']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['lr_psame']['steps'], data_fc2net['test']['regular']['lr_psame']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('LR psame score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test LR psame scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_dnn_lr_psame'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['lr_psame']['steps']   , data_wrn['train']['regular']['lr_psame']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['lr_psame']['steps'] , data_lenet['train']['regular']['lr_psame']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['lr_psame']['steps'], data_fc2net['train']['regular']['lr_psame']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('LR psame score')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train LR psame scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_dnn_lr_psame'))
    plt.close()

    # Confidence
    # DNN
    plt.figure()
    plt.plot(data_wrn['test']['regular']['dnn_confidence_avg']['steps']   , data_wrn['test']['regular']['dnn_confidence_avg']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['dnn_confidence_avg']['steps'] , data_lenet['test']['regular']['dnn_confidence_avg']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['dnn_confidence_avg']['steps'], data_fc2net['test']['regular']['dnn_confidence_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('DNN confidence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test DNN confidence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_dnn_confidence'))
    plt.close()


    plt.figure()
    plt.plot(data_wrn['train']['regular']['dnn_confidence_avg']['steps']   , data_wrn['train']['regular']['dnn_confidence_avg']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['dnn_confidence_avg']['steps'] , data_lenet['train']['regular']['dnn_confidence_avg']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['dnn_confidence_avg']['steps'], data_fc2net['train']['regular']['dnn_confidence_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('DNN confidence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train DNN confidence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_dnn_confidence'))
    plt.close()

    #KNN
    plt.figure()
    plt.plot(data_wrn['test']['regular']['knn_confidence_avg']['steps']   , data_wrn['test']['regular']['knn_confidence_avg']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['knn_confidence_avg']['steps'] , data_lenet['test']['regular']['knn_confidence_avg']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['knn_confidence_avg']['steps'], data_fc2net['test']['regular']['knn_confidence_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KNN confidence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test KNN confidence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_knn_confidence'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['knn_confidence_avg']['steps']   , data_wrn['train']['regular']['knn_confidence_avg']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['knn_confidence_avg']['steps'] , data_lenet['train']['regular']['knn_confidence_avg']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['knn_confidence_avg']['steps'], data_fc2net['train']['regular']['knn_confidence_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KNN confidence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train KNN confidence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_knn_confidence'))
    plt.close()

    #SVM
    plt.figure()
    plt.plot(data_wrn['test']['regular']['svm_confidence_avg']['steps']   , data_wrn['test']['regular']['svm_confidence_avg']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['svm_confidence_avg']['steps'] , data_lenet['test']['regular']['svm_confidence_avg']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['svm_confidence_avg']['steps'], data_fc2net['test']['regular']['svm_confidence_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('SVM confidence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test SVM confidence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_svm_confidence'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['svm_confidence_avg']['steps']   , data_wrn['train']['regular']['svm_confidence_avg']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['svm_confidence_avg']['steps'] , data_lenet['train']['regular']['svm_confidence_avg']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['svm_confidence_avg']['steps'], data_fc2net['train']['regular']['svm_confidence_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('SVM confidence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train SVM confidence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_svm_confidence'))
    plt.close()


    # LR
    plt.figure()
    plt.plot(data_wrn['test']['regular']['lr_confidence_avg']['steps']   , data_wrn['test']['regular']['lr_confidence_avg']['values']   , 'r')
    plt.plot(data_lenet['test']['regular']['lr_confidence_avg']['steps'] , data_lenet['test']['regular']['lr_confidence_avg']['values'] , 'g')
    plt.plot(data_fc2net['test']['regular']['lr_confidence_avg']['steps'], data_fc2net['test']['regular']['lr_confidence_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('LR confidence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Test LR confidence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'test_lr_confidence'))
    plt.close()

    plt.figure()
    plt.plot(data_wrn['train']['regular']['lr_confidence_avg']['steps']   , data_wrn['train']['regular']['lr_confidence_avg']['values']   , 'r')
    plt.plot(data_lenet['train']['regular']['lr_confidence_avg']['steps'] , data_lenet['train']['regular']['lr_confidence_avg']['values'] , 'g')
    plt.plot(data_fc2net['train']['regular']['lr_confidence_avg']['steps'], data_fc2net['train']['regular']['lr_confidence_avg']['values'], 'b')
    plt.gca().yaxis.grid(True)
    plt.ylabel('LR confidence')
    plt.legend(['Wide Resnet 28-10', 'LeNet', 'MLP-640'])
    plt.title('Train LR confidence scores on {}'.format(dataset))
    plt.savefig(os.path.join(plot_directory, 'train_lr_confidence'))
    plt.close()
