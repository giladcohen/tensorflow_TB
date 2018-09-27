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

root_dirs = ['/data/gilad/logs/metrics/wrn/mnist/log_0049_270818_metrics_w_confidence-SUPERSEED=27081800',
             '/data/gilad/logs/metrics/wrn/mnist/random/log_0333_250918_metrics_longer-SUPERSEED=25091800',
             '/data/gilad/logs/metrics/wrn/cifar10/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800',
             '/data/gilad/logs/metrics/wrn/cifar10/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800',
             '/data/gilad/logs/metrics/wrn/cifar100/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800',
             '/data/gilad/logs/metrics/wrn/cifar100/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800',

             '/data/gilad/logs/metrics/lenet/mnist/log_0152_140918_metrics-SUPERSEED=14091800',
             '/data/gilad/logs/metrics/lenet/mnist/random/log_0333_250918_metrics_longer-SUPERSEED=25091800',
             '/data/gilad/logs/metrics/lenet/cifar10/log_1319_120918_metrics-SUPERSEED=12091800',
             '/data/gilad/logs/metrics/lenet/cifar10/random/log_1319_120918_metrics-SUPERSEED=12091800',
             '/data/gilad/logs/metrics/lenet/cifar100/log_1319_120918_metrics-SUPERSEED=12091800',
             '/data/gilad/logs/metrics/lenet/cifar100/random/log_1319_120918_metrics-SUPERSEED=12091800',

             '/data/gilad/logs/metrics/fc2net/mnist/log_0709_150918_metrics-SUPERSEED=15091800',
             '/data/gilad/logs/metrics/fc2net/mnist/random/log_0333_250918_metrics_longer-SUPERSEED=25091800',
             '/data/gilad/logs/metrics/fc2net/cifar10/log_1025_150918_metrics-SUPERSEED=15091800',
             '/data/gilad/logs/metrics/fc2net/cifar10/random/log_1025_150918_metrics-SUPERSEED=15091800',
             '/data/gilad/logs/metrics/fc2net/cifar100/log_1025_150918_metrics-SUPERSEED=15091800',
             '/data/gilad/logs/metrics/fc2net/cifar100/random/log_1025_150918_metrics-SUPERSEED=15091800'
             ]

plot_root = '/data/gilad/logs/metrics/all_plots/plots'

for root_dir in root_dirs:
    if 'cifar100' in root_dir:
        dataset = 'cifar100'
    elif 'cifar10' in root_dir:
        dataset = 'cifar10'
    else:
        dataset = 'mnist'
    if 'random' in root_dir:
        dataset = 'random_' + dataset

    # set plot dir
    if 'wrn' in root_dir:
        plot_directory = os.path.join(plot_root, 'wrn')
    elif 'lenet' in root_dir:
        plot_directory = os.path.join(plot_root, 'lenet')
    else:
        plot_directory = os.path.join(plot_root, 'mlp640')
    plot_directory = os.path.join(plot_directory, dataset)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data = json.load(f)

    # regular plots
    # ploting the models' scores
    plt.figure()
    plt.plot(data['test']['regular']['dnn_score']['steps'], data['test']['regular']['dnn_score']['values'], 'r')
    plt.plot(data['test']['regular']['knn_score']['steps'], data['test']['regular']['knn_score']['values'], 'b')
    plt.plot(data['test']['regular']['svm_score']['steps'], data['test']['regular']['svm_score']['values'], 'k')
    plt.plot(data['test']['regular']['lr_score']['steps'] , data['test']['regular']['lr_score']['values'] , 'g')
    plt.gca().yaxis.grid(True)
    plt.ylim(0.0, 1.0)
    plt.ylabel('accuracy score')
    plt.legend(['dnn', 'knn', 'svm', 'lr'])
    plt.title('Test accuracy scores on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_accuracy_score.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['regular']['dnn_score']['steps'], data['train']['regular']['dnn_score']['values'], 'r')
    plt.plot(data['train']['regular']['knn_score']['steps'], data['train']['regular']['knn_score']['values'], 'b')
    plt.plot(data['train']['regular']['svm_score']['steps'], data['train']['regular']['svm_score']['values'], 'k')
    plt.plot(data['train']['regular']['lr_score']['steps'] , data['train']['regular']['lr_score']['values'] , 'g')
    plt.gca().yaxis.grid(True)
    plt.ylim(0.0, 1.0)
    plt.ylabel('accuracy score')
    plt.legend(['dnn', 'knn', 'svm', 'lr'])
    plt.title('Train accuracy scores on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_accuracy_score.png'))
    plt.close()

    # ploting the psame score (correspondence to DNN)
    plt.figure()
    plt.plot(data['test']['regular']['knn_psame']['steps'], data['test']['regular']['knn_psame']['values'], 'b')
    plt.plot(data['test']['regular']['svm_psame']['steps'], data['test']['regular']['svm_psame']['values'], 'k')
    plt.plot(data['test']['regular']['lr_psame']['steps'] , data['test']['regular']['lr_psame']['values'] , 'g')
    plt.gca().yaxis.grid(True)
    plt.ylim(0.0, 1.0)
    plt.ylabel('psame to DNN')
    plt.legend(['knn', 'svm', 'lr'])
    plt.title('Test psame on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_psame.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['regular']['knn_psame']['steps'], data['train']['regular']['knn_psame']['values'], 'b')
    plt.plot(data['train']['regular']['svm_psame']['steps'], data['train']['regular']['svm_psame']['values'], 'k')
    plt.plot(data['train']['regular']['lr_psame']['steps'] , data['train']['regular']['lr_psame']['values'] , 'g')
    plt.gca().yaxis.grid(True)
    plt.ylim(0.0, 1.0)
    plt.ylabel('psame to DNN')
    plt.legend(['knn', 'svm', 'lr'])
    plt.title('Train psame on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_psame.png'))
    plt.close()

    # plotting confidences avg
    plt.figure()
    plt.plot(data['test']['regular']['dnn_confidence_avg']['steps'], data['test']['regular']['dnn_confidence_avg']['values'], 'r')
    plt.plot(data['test']['regular']['knn_confidence_avg']['steps'], data['test']['regular']['knn_confidence_avg']['values'], 'b')
    plt.plot(data['test']['regular']['svm_confidence_avg']['steps'], data['test']['regular']['svm_confidence_avg']['values'], 'k')
    plt.plot(data['test']['regular']['lr_confidence_avg']['steps'] , data['test']['regular']['lr_confidence_avg']['values'] , 'g')
    plt.gca().yaxis.grid(True)
    plt.ylim(0.0, 1.0)
    plt.ylabel('confidence avg')
    plt.legend(['dnn', 'knn', 'svm', 'lr'])
    plt.title('Test confidence (avg) on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_confidence_avg.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['regular']['dnn_confidence_avg']['steps'], data['train']['regular']['dnn_confidence_avg']['values'], 'r')
    plt.plot(data['train']['regular']['knn_confidence_avg']['steps'], data['train']['regular']['knn_confidence_avg']['values'], 'b')
    plt.plot(data['train']['regular']['svm_confidence_avg']['steps'], data['train']['regular']['svm_confidence_avg']['values'], 'k')
    plt.plot(data['train']['regular']['lr_confidence_avg']['steps'] , data['train']['regular']['lr_confidence_avg']['values'] , 'g')
    plt.gca().yaxis.grid(True)
    plt.ylim(0.0, 1.0)
    plt.ylabel('confidence avg')
    plt.legend(['dnn', 'knn', 'svm', 'lr'])
    plt.title('Train confidence (avg) on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_confidence_avg.png'))
    plt.close()

    # plotting confidences median
    plt.figure()
    plt.plot(data['test']['regular']['dnn_confidence_median']['steps'], data['test']['regular']['dnn_confidence_median']['values'], 'r')
    plt.plot(data['test']['regular']['knn_confidence_median']['steps'], data['test']['regular']['knn_confidence_median']['values'], 'b')
    plt.plot(data['test']['regular']['svm_confidence_median']['steps'], data['test']['regular']['svm_confidence_median']['values'], 'k')
    plt.plot(data['test']['regular']['lr_confidence_median']['steps'] , data['test']['regular']['lr_confidence_median']['values'] , 'g')
    plt.gca().yaxis.grid(True)
    plt.ylim(0.0, 1.0)
    plt.ylabel('confidence median')
    plt.legend(['dnn', 'knn', 'svm', 'lr'])
    plt.title('Test confidence (median) on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_confidence_median.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['regular']['dnn_confidence_median']['steps'], data['train']['regular']['dnn_confidence_median']['values'], 'r')
    plt.plot(data['train']['regular']['knn_confidence_median']['steps'], data['train']['regular']['knn_confidence_median']['values'], 'b')
    plt.plot(data['train']['regular']['svm_confidence_median']['steps'], data['train']['regular']['svm_confidence_median']['values'], 'k')
    plt.plot(data['train']['regular']['lr_confidence_median']['steps'] , data['train']['regular']['lr_confidence_median']['values'] , 'g')
    plt.gca().yaxis.grid(True)
    plt.ylim(0.0, 1.0)
    plt.ylabel('confidence median')
    plt.legend(['dnn', 'knn', 'svm', 'lr'])
    plt.title('Train confidence (median) on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_confidence_median.png'))
    plt.close()

    # plotting KL div
    plt.figure()
    plt.plot(data['test']['regular']['knn_kl_div_avg']['steps'], data['test']['regular']['knn_kl_div_avg']['values'], 'b')
    plt.plot(data['test']['regular']['lr_kl_div_avg']['steps'] , data['test']['regular']['lr_kl_div_avg']['values'] , 'g')
    plt.plot(data['test']['regular']['svm_kl_div_avg']['steps'], data['test']['regular']['svm_kl_div_avg']['values'], 'k')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KL div1 to DNN')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Test KL div1 avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div1.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['regular']['knn_kl_div_avg']['steps'], data['train']['regular']['knn_kl_div_avg']['values'], 'b')
    plt.plot(data['train']['regular']['lr_kl_div_avg']['steps'] , data['train']['regular']['lr_kl_div_avg']['values'] , 'g')
    plt.plot(data['train']['regular']['svm_kl_div_avg']['steps'], data['train']['regular']['svm_kl_div_avg']['values'], 'k')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KL div1 to DNN')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Train KL div1 avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_kl_div1.png'))
    plt.close()

    # plotting KL div2
    plt.figure()
    plt.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], data['test']['regular']['knn_kl_div2_avg']['values'], 'b')
    plt.plot(data['test']['regular']['lr_kl_div2_avg']['steps'] , data['test']['regular']['lr_kl_div2_avg']['values'] , 'g')
    plt.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], data['test']['regular']['svm_kl_div2_avg']['values'], 'k')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KL div2 to DNN')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Test KL div2 avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div2.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], data['train']['regular']['knn_kl_div2_avg']['values'], 'b')
    plt.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , data['train']['regular']['lr_kl_div2_avg']['values'] , 'g')
    plt.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], data['train']['regular']['svm_kl_div2_avg']['values'], 'k')
    plt.gca().yaxis.grid(True)
    plt.ylabel('KL div2 to DNN')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Train KL div1 avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_kl_div2.png'))
    plt.close()

    # Layer printing
    if 'wrn' in root_dir:
        layers = ['input\nimages', 'init\nconv',
                  'unit_1_1', 'unit_1_2', 'unit_1_3', 'unit_1_4',
                  'unit_2_1', 'unit_2_2', 'unit_2_3', 'unit_2_4',
                  'unit_3_1', 'unit_3_2', 'unit_3_3', 'unit_3_4',
                  'embedding\nvector']
    elif 'lenet' in root_dir:
        layers = ['input\nimages', 'init\nconv', 'conv1', 'pool1', 'conv2', 'pool2', 'embedding\nvector']
    else:
        continue

    x = np.arange(len(layers))

    # scores
    plt.figure()
    plt.plot(data['test']['layer']['knn_score'], 'b')
    plt.plot(data['test']['layer']['lr_score'] , 'g')
    plt.plot(data['test']['layer']['svm_score'], 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('accuracy score')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Test accuracy score on {})'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_accuracy_score_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['knn_score'], 'b')
    plt.plot(data['train']['layer']['lr_score'] , 'g')
    plt.plot(data['train']['layer']['svm_score'], 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('accuracy score')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Train accuracy score on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_accuracy_score_vs_layer.png'))
    plt.close()

    # psame scores
    plt.figure()
    plt.plot(data['test']['layer']['svm_knn_psame'], 'b')
    plt.plot(data['test']['layer']['svm_lr_psame'] , 'g')
    plt.plot(data['test']['layer']['lr_knn_psame'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('psame score')
    plt.legend(['svm-knn', 'svm-lr', 'lr-knn'])
    plt.title('Test psame score on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_psame_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['svm_knn_psame'], 'b')
    plt.plot(data['train']['layer']['svm_lr_psame'] , 'g')
    plt.plot(data['train']['layer']['lr_knn_psame'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('psame score')
    plt.legend(['svm-knn', 'svm-lr', 'lr-knn'])
    plt.title('Train psame score on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_psame_vs_layer.png'))
    plt.close()

    # confidences
    plt.figure()
    plt.plot(data['test']['layer']['knn_confidence_avg'], 'b')
    plt.plot(data['test']['layer']['lr_confidence_avg'] , 'g')
    plt.plot(data['test']['layer']['svm_confidence_avg'], 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('confidence avg')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Test confidence avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_confidence_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['knn_confidence_avg'], 'b')
    plt.plot(data['train']['layer']['lr_confidence_avg'] , 'g')
    plt.plot(data['train']['layer']['svm_confidence_avg'], 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('confidence avg')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Train confidence avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_confidence_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['test']['layer']['knn_confidence_median'], 'b')
    plt.plot(data['test']['layer']['lr_confidence_median'] , 'g')
    plt.plot(data['test']['layer']['svm_confidence_median'], 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('confidence median')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Test confidence median on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_confidence_median_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['knn_confidence_median'], 'b')
    plt.plot(data['train']['layer']['lr_confidence_median'] , 'g')
    plt.plot(data['train']['layer']['svm_confidence_median'], 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('confidence median')
    plt.legend(['knn', 'lr', 'svm'])
    plt.title('Train confidence median on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_confidence_median_vs_layer.png'))
    plt.close()

    # KL divergences
    plt.figure()
    plt.plot(data['test']['layer']['lr_knn_kl_div_avg'] , 'b')
    plt.plot(data['test']['layer']['svm_knn_kl_div_avg'], 'g')
    plt.plot(data['test']['layer']['svm_lr_kl_div_avg'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div_avg')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Test kl_div_avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['lr_knn_kl_div_avg'] , 'b')
    plt.plot(data['train']['layer']['svm_knn_kl_div_avg'], 'g')
    plt.plot(data['train']['layer']['svm_lr_kl_div_avg'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div_avg')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Train kl_div_avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['test']['layer']['lr_knn_kl_div2_avg'] , 'b')
    plt.plot(data['test']['layer']['svm_knn_kl_div2_avg'], 'g')
    plt.plot(data['test']['layer']['svm_lr_kl_div2_avg'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div2_avg')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Test kl_div2_avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div2_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['lr_knn_kl_div2_avg'] , 'b')
    plt.plot(data['train']['layer']['svm_knn_kl_div2_avg'], 'g')
    plt.plot(data['train']['layer']['svm_lr_kl_div2_avg'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div2_avg')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Train kl_div2_avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_kl_div2_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['test']['layer']['lr_knn_kl_div3_avg'] , 'b')
    plt.plot(data['test']['layer']['svm_knn_kl_div3_avg'], 'g')
    plt.plot(data['test']['layer']['svm_lr_kl_div3_avg'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div3_avg')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Test kl_div3_avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div3_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['lr_knn_kl_div3_avg'] , 'b')
    plt.plot(data['train']['layer']['svm_knn_kl_div3_avg'], 'g')
    plt.plot(data['train']['layer']['svm_lr_kl_div3_avg'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div3_avg')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Train kl_div3_avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_kl_div3_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['test']['layer']['lr_knn_kl_div4_avg'] , 'b')
    plt.plot(data['test']['layer']['svm_knn_kl_div4_avg'], 'g')
    plt.plot(data['test']['layer']['svm_lr_kl_div4_avg'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div4_avg')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Test kl_div4_avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div4_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['lr_knn_kl_div4_avg'] , 'b')
    plt.plot(data['train']['layer']['svm_knn_kl_div4_avg'], 'g')
    plt.plot(data['train']['layer']['svm_lr_kl_div4_avg'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div4_avg')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Train kl_div4_avg on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_kl_div4_avg_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['test']['layer']['lr_knn_kl_div3_median'] , 'b')
    plt.plot(data['test']['layer']['svm_knn_kl_div3_median'], 'g')
    plt.plot(data['test']['layer']['svm_lr_kl_div3_median'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div3_median')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Test kl_div3_median on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div3_median_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['lr_knn_kl_div3_median'] , 'b')
    plt.plot(data['train']['layer']['svm_knn_kl_div3_median'], 'g')
    plt.plot(data['train']['layer']['svm_lr_kl_div3_median'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div3_median')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Train kl_div3_median on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_kl_div3_median_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['test']['layer']['lr_knn_kl_div4_median'] , 'b')
    plt.plot(data['test']['layer']['svm_knn_kl_div4_median'], 'g')
    plt.plot(data['test']['layer']['svm_lr_kl_div4_median'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div4_median')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Test kl_div4_median on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'test_kl_div4_median_vs_layer.png'))
    plt.close()

    plt.figure()
    plt.plot(data['train']['layer']['lr_knn_kl_div4_median'] , 'b')
    plt.plot(data['train']['layer']['svm_knn_kl_div4_median'], 'g')
    plt.plot(data['train']['layer']['svm_lr_kl_div4_median'] , 'k')
    plt.xticks(x)
    plt.gca().set_xticklabels(layers, fontdict={'rotation': 'vertical'})
    plt.gca().yaxis.grid(True)
    plt.ylabel('kl_div4_median')
    plt.legend(['lr-knn', 'svm-knn', 'svm-lr'])
    plt.title('Train kl_div4_median on {}'.format(dataset))
    #plt.show()
    plt.savefig(os.path.join(plot_directory, 'train_kl_div4_median_vs_layer.png'))
    plt.close()
