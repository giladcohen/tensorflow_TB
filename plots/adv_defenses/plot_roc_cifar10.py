from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import os
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from lid_adversarial_subspace_detection.util import (random_split, block_split, train_lr, compute_roc)
import matplotlib.pyplot as plt
from tensorflow_TB.utils.plots import add_subplot_axes
import matplotlib.patches as patches

def load_characteristics(characteristics_file):
    X, Y = None, None
    data = np.load(characteristics_file)
    if X is None:
        X = data[:, :-1]
    if Y is None:
        Y = data[:, -1]  # labels only need to load once

    return X, Y

def calc_metrics(train_characteristics_file, test_characteristics_file):
    # X, Y = load_characteristics(characteristics_file)
    X_train, Y_train = load_characteristics(train_characteristics_file)
    X_test, Y_test   = load_characteristics(test_characteristics_file)

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    print("Train data size: ", X_train.shape)
    print("Test data size: ", X_test.shape)

    ## Build detector
    lr = train_lr(X_train, Y_train)

    ## Evaluate detector
    y_pred       = lr.predict_proba(X_test)[:, 1]
    y_label_pred = lr.predict(X_test)

    # AUC
    fpr, tpr, auc_score = compute_roc(Y_test, y_pred, plot=False)
    precision = precision_score(Y_test, y_label_pred)
    recall = recall_score(Y_test, y_label_pred)
    acc = accuracy_score(Y_test, y_label_pred)
    print('Detector ROC-AUC score: {}, accuracy: {}, precision: {}, recall: {}'.format(auc_score, acc, precision, recall))

    return fpr, tpr, auc_score, acc

plt.rcParams['interactive'] = False
subpos = np.array([0.35, 0.25, 0.5, 0.4])
fig = plt.figure(figsize=(12.0, 5.0))

# DeepFool
ax1 = fig.add_subplot(121)
# DKNN
train_characteristics_file = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/deepfool/dknn/k_4800_train.npy'
test_characteristics_file  = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/deepfool/dknn/k_4800_train.npy'
fpr1, tpr1, auc_score, acc = calc_metrics(train_characteristics_file, test_characteristics_file)
ax1.plot(fpr1, tpr1, 'k--')

# LID
train_characteristics_file = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/deepfool/lid/k_17_batch_100_train.npy'
test_characteristics_file  = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/deepfool/lid/k_17_batch_100_test.npy'
fpr2, tpr2, auc_score, acc = calc_metrics(train_characteristics_file, test_characteristics_file)
ax1.plot(fpr2, tpr2, 'b-.')

# Mahalanobis
train_characteristics_file = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/deepfool/mahalanobis/magnitude_0.0001_scale_1.0_train.npy'
test_characteristics_file  = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/deepfool/mahalanobis/magnitude_0.0001_scale_1.0_test.npy'
fpr3, tpr3, auc_score, acc = calc_metrics(train_characteristics_file, test_characteristics_file)
ax1.plot(fpr3, tpr3, 'g:', linewidth=2)

# NNIF
train_characteristics_file = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/deepfool/nnif/max_indices_200_train.npy'
test_characteristics_file  = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/deepfool/nnif/max_indices_200_test.npy'
fpr4, tpr4, auc_score, acc = calc_metrics(train_characteristics_file, test_characteristics_file)
ax1.plot(fpr4, tpr4, 'r')

ax1.set_title('Deepfool')
ax1.grid()
ax1.set_xlabel('FPR', labelpad=5, fontdict={'fontsize': 12})
ax1.set_ylabel('TPR', labelpad=5, fontdict={'fontsize': 12})
ax1.legend(['D$k$NN', 'LID', 'Mahalanobis', 'NNIF (ours)'], loc=(0.7, 0.05))

subax1 = add_subplot_axes(ax1, subpos + [-0.25, 0.18, 0, 0])
subax1.set_xlim([-0.02, 0.5])
subax1.set_ylim([0.8, 1.05])
subax1.set_yticks([0.85, 0.9, 0.95, 1.0])
subax1.plot(fpr1, tpr1, 'k--')
subax1.plot(fpr2, tpr2, 'b-.')
subax1.plot(fpr3, tpr3, 'g:', linewidth=2)
subax1.plot(fpr4, tpr4, 'r')
ax1.add_patch(patches.Polygon(xy=np.array([[0.0, 1.01], [0, 0.8], [0.213, 0.383], [0.66, 0.81], [0.5, 1.01]]), closed=True, color='silver'))

# CW
ax2 = fig.add_subplot(122)
# DKNN
train_characteristics_file = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/cw_targeted/dknn/k_4700_train.npy'
test_characteristics_file  = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/cw_targeted/dknn/k_4700_train.npy'
fpr1, tpr1, auc_score, acc = calc_metrics(train_characteristics_file, test_characteristics_file)
ax2.plot(fpr1, tpr1, 'k--')

# LID
train_characteristics_file = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/cw_targeted/lid/k_20_batch_100_train.npy'
test_characteristics_file  = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/cw_targeted/lid/k_20_batch_100_test.npy'
fpr2, tpr2, auc_score, acc = calc_metrics(train_characteristics_file, test_characteristics_file)
ax2.plot(fpr2, tpr2, 'b-.')

# Mahalanobis
train_characteristics_file = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/cw_targeted/mahalanobis/magnitude_8e-05_scale_1.0_train.npy'
test_characteristics_file  = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/cw_targeted/mahalanobis/magnitude_8e-05_scale_1.0_test.npy'
fpr3, tpr3, auc_score, acc = calc_metrics(train_characteristics_file, test_characteristics_file)
ax2.plot(fpr3, tpr3, 'g:', linewidth=2)

# NNIF
train_characteristics_file = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/cw_targeted/nnif/max_indices_200_train.npy'
test_characteristics_file  = '/data/gilad/logs/influence/cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000/cw_targeted/nnif/max_indices_200_test.npy'
fpr4, tpr4, auc_score, acc = calc_metrics(train_characteristics_file, test_characteristics_file)
ax2.plot(fpr4, tpr4, 'r')

ax2.set_title('Carlini-Wagner')
ax2.grid()
ax2.set_xlabel('FPR', labelpad=5, fontdict={'fontsize': 12})
ax2.set_ylabel('TPR', labelpad=5, fontdict={'fontsize': 12})
ax2.legend(['D$k$NN', 'LID', 'Mahalanobis', 'NNIF (ours)'], loc=(0.7, 0.05))

subax2 = add_subplot_axes(ax2, subpos + [-0.05, 0.18, 0, 0])
subax2.set_xlim([-0.02, 0.5])
subax2.set_ylim([0.8, 1.05])
subax2.set_yticks([0.85, 0.9, 0.95, 1.0])
subax2.plot(fpr1, tpr1, 'k--')
subax2.plot(fpr2, tpr2, 'b-.')
subax2.plot(fpr3, tpr3, 'g:', linewidth=2)
subax2.plot(fpr4, tpr4, 'r')
ax2.add_patch(patches.Polygon(xy=np.array([[0.0, 1.01], [0, 0.8], [0.213, 0.383], [0.66, 0.81], [0.5, 1.01]]), closed=True, color='silver'))

plt.tight_layout()
plt.savefig('auc_cifar10.png', dpi=350)


