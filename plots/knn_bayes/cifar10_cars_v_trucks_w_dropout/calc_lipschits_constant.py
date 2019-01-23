from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json
from scipy.spatial import distance_matrix
import tensorflow as tf
import argparse

# NORM = 'L2'
# PERCENTAGE = 0.5
# INPUT = 'image'  # 'image' or 'embedding'

def calc_D_vec(sf_vec):
    return sf_vec[:, 1] - sf_vec[:, 0]

def get_raw_data():
    """
    output raw data
    :return: np.ndarrays (X_test, y_test)
    """
    data = tf.keras.datasets.cifar10
    (_, _), (X_test, y_test) = data.load_data()
    test_indices = []
    for cls in [1, 9]:
        test_indices += np.where(y_test == cls)[0].tolist()
    test_indices.sort()
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]

    # replace label 1->0 and 9->1
    for i, label in enumerate(y_test):
        if label == 1:
            y_test[i] = 0
        elif label == 9:
            y_test[i] = 1
        else:
            err_str = 'y_test[{}] equals {} instead of 1 or 9'.format(i, label)
            raise AssertionError(err_str)

    y_test = np.squeeze(y_test, axis=1)
    #     X_test = np.expand_dims(X_test, axis=-1)

    return (X_test, y_test)

def print_c_lipshits(NORM, PERCENTAGE, INPUT, n):
    assert NORM in ['L1', 'L2']
    assert 0.0 < PERCENTAGE <= 100.0
    assert INPUT in ['image', 'embedding']

    # an ugly way to get the test data again, this should be done as above. Relevant for all n
    (X_test, y_test_ref) = get_raw_data()

    print("Start working on norm={}, percentage={}, input={}, n={}".format(NORM, PERCENTAGE, INPUT, n))
    logdir = '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=21011900'.format(n)
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

    assert (y_test == y_test_ref).all(), 'y_test_ref must match y_test'

    train_size = y_train.shape[0]
    test_size  = y_test.shape[0]

    # calculating the D vectors
    D_train = calc_D_vec(train_dnn_predictions_prob)
    D_test  = calc_D_vec(test_dnn_predictions_prob)

    # taking only a small subset for debug
    # test_size  = 200
    # indices = np.random.choice(range(test_size), 200, replace=False)
    # X_test_features           = X_test_features[indices]
    # y_test                    = y_test[indices]
    # test_dnn_predictions_prob = test_dnn_predictions_prob[indices]
    # D_test                    = D_test[indices]

    # creating features mat and D mat
    if INPUT == 'image':
        X_test_2D_vec = X_test.reshape(X_test.shape[0], -1)
        features_mat = distance_matrix(X_test_2D_vec, X_test_2D_vec, int(NORM[-1]))
    else:
        features_mat = distance_matrix(X_test_features, X_test_features, int(NORM[-1]))
    D_mat = np.subtract.outer(D_test, D_test)
    D_mat = np.abs(D_mat)

    # aggregating all distances
    cnt = 0
    all_feature_distances = []
    for i in range(0, test_size):
        for j in range(i+1, test_size):
            all_feature_distances.append(features_mat[i, j])
            cnt += 1
    assert cnt == ((test_size - 1)*test_size / 2)
    all_feature_distances = np.array(all_feature_distances)
    all_feature_distances.sort()

    # I want to take just PERCENTAGE of the distances. therefore...
    if PERCENTAGE == 100.0:
        index = -1
    else:
        index = int(all_feature_distances.shape[0] * PERCENTAGE / 100)
    max_dist = all_feature_distances[index]

    # num_bins = int(cnt/100)  # I want just one percent of the distances
    # dist_hist, bin_edges = np.histogram(all_feature_distances, num_bins)
    # max_dist = bin_edges[1]

    # for every ||x-y|| < max_dist, we need to find |D(x)-D(z)|
    all_feature_distances = []
    all_D_distances       = []
    for i in range(0, test_size):
        for j in range(i+1, test_size):
            if features_mat[i, j] <= max_dist:
                all_feature_distances.append(features_mat[i, j])
                all_D_distances.append(D_mat[i, j])

    # plot the scatter plot
    all_feature_distances = np.array(all_feature_distances)
    all_D_distances       = np.array(all_D_distances)
    D_div_xz              = all_D_distances/all_feature_distances
    C_Lipschits = np.max(D_div_xz)
    plt.scatter(all_feature_distances, D_div_xz, s=0.5)
    plt.ylim([0, 1.1*C_Lipschits])
    plt.xlabel('||x-z||')
    plt.ylabel('|D(x)-D(z)|/||x-z||')
    plt.title('|D(x)-D(z)|/||x-z|| measure for norm {}, input {}, percentage {}, n={}'.format(NORM, INPUT, PERCENTAGE, n))
    dir = os.path.join(os.path.dirname(__file__), 'norm_{}/input_{}/percentage_{}/n_{}'.format(NORM, INPUT, PERCENTAGE, n))
    if not os.path.exists(dir):
        os.makedirs(dir)
    # plt.savefig(os.path.join(dir, 'C_Lipschits_C={:0.5f}.png'.format(C_Lipschits)))
    plt.savefig(os.path.join(dir, 'C_Lipschits_C={}.png'.format(C_Lipschits)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--NORM'              , action='store')
    parser.add_argument('--PERCENTAGE'        , action='store')
    parser.add_argument('--INPUT'             , action='store')
    parser.add_argument('--n'                 , action='store')
    args = parser.parse_args()

    print_c_lipshits(args.NORM, float(args.PERCENTAGE), args.INPUT, int(args.n))

    print('script done')

