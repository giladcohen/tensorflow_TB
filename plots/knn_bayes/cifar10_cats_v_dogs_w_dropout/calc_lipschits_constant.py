from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

norm = 'L2'

def D(x):
    return x[1] - x[0]

logdir = '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_1k-SUPERSEED=08011900'
test_dir = os.path.join(logdir, 'test')
train_features_file             = os.path.join(test_dir, 'train_features.npy')
test_features_file              = os.path.join(test_dir, 'test_features.npy')
train_dnn_predictions_prob_file = os.path.join(test_dir, 'train_dnn_predictions_prob.npy')
test_dnn_predictions_prob_file  = os.path.join(test_dir, 'test_dnn_predictions_prob.npy')
train_labels_file               = os.path.join(test_dir, 'train_labels.npy')
test_labels_file                = os.path.join(test_dir, 'test_labels.npy')

X_train_features                = np.load(train_features_file)
y_train                         = np.load(train_labels_file)
train_dnn_predictions_prob      = np.load(train_dnn_predictions_prob_file)
X_test_features                 = np.load(test_features_file)
y_test                          = np.load(test_labels_file)
test_dnn_predictions_prob       = np.load(test_dnn_predictions_prob_file)

# creating features mat
features_mat = np.empty(shape=(y_test.shape[0], y_test.shape[0]))



