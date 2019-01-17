from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json
from scipy.spatial import distance_matrix
norm = 'L2'

def calc_D_vec(sf_vec):
    return sf_vec[:, 1] - sf_vec[:, 0]

n = 10
logdir = '/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=08011900'.format(n)
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
features_mat = distance_matrix(X_test_features, X_test_features, int(norm[-1]))
D_mat = np.subtract.outer(D_test, D_test)
D_mat = np.abs(D_mat)

# calculating a histogram for
cnt = 0
all_feature_distances = []
for i in range(0, test_size):
    for j in range(i+1, test_size):
        all_feature_distances.append(features_mat[i, j])
        cnt += 1
assert cnt == ((test_size - 1)*test_size / 2)
all_feature_distances = np.array(all_feature_distances)
all_feature_distances.sort()

# I want to take just the 0.05% of the distances. therefore...
index = all_feature_distances.shape[0] // 200
max_embedded_dist = all_feature_distances[index]

# num_bins = int(cnt/100)  # I want just one percent of the distances
# dist_hist, bin_edges = np.histogram(all_feature_distances, num_bins)
# max_embedded_dist = bin_edges[1]

# for every ||x-y|| < max_embedded_dist, we need to find |D(x)-D(z)|
all_feature_distances = []
all_D_distances       = []
for i in range(0, test_size):
    for j in range(i+1, test_size):
        if features_mat[i, j] < max_embedded_dist:
            all_feature_distances.append(features_mat[i, j])
            all_D_distances.append(D_mat[i, j])

# plot the scatter plot
all_feature_distances = np.array(all_feature_distances)
all_D_distances       = np.array(all_D_distances)
D_div_xz              = all_D_distances/all_feature_distances
plt.scatter(all_feature_distances, D_div_xz, s=0.5)
plt.xlabel('||x-z||')
plt.ylabel('|D(x)-D(z)|/||x-z||')
C_Lipschits = np.max(D_div_xz)
plt.savefig('C_Lipschits_n={}_C={}.png'.format(n, C_Lipschits))
