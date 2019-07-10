from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
import platform
# Force matplotlib to not use any Xwindows backend.
if platform.system() == 'Linux':
    matplotlib.use('Agg')

import os
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from lid_adversarial_subspace_detection.util import (random_split, block_split, train_lr, compute_roc)

from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cifar10', 'dataset: cifar10/100 or svhn')
flags.DEFINE_string('attack', 'cw', 'adversarial attack: deepfool, jsma, cw')
flags.DEFINE_bool('targeted', True, 'whether or not the adversarial attack is targeted')
flags.DEFINE_string('characteristics', 'lid', 'type of defence: lid/mahalanobis/dknn/nnif')
flags.DEFINE_bool('with_noise', False, 'whether or not to include noisy samples')

# FOR LID
flags.DEFINE_integer('k_nearest', 17, 'number of nearest neighbors to use for LID/DkNN detection')

# FOR MAHANABOLIS
flags.DEFINE_float('magnitude', 0.002, 'magnitude for mahalanobis detection')
flags.DEFINE_float('rgb_scale', 1, 'scale for mahalanobis')

# FOR NNIF
flags.DEFINE_integer('max_indices', 200, 'maximum number of helpful indices to use in NNIF detection')
flags.DEFINE_string('ablation', '1111', 'for ablation test')



if FLAGS.dataset == 'cifar10':
    CHECKPOINT_NAME = 'cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000'
elif FLAGS.dataset == 'cifar100':
    CHECKPOINT_NAME = 'cifar100/log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01'
elif FLAGS.dataset == 'svhn':
    CHECKPOINT_NAME = 'svhn/log_120519_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1257'
else:
    raise AssertionError('dataset {} not supported'.format(FLAGS.dataset))

model_dir          = os.path.join('/data/gilad/logs/influence', CHECKPOINT_NAME)
attack_dir         = os.path.join(model_dir, FLAGS.attack)
if FLAGS.targeted:
    attack_dir = attack_dir + '_targeted'
characteristics_dir = os.path.join(attack_dir, FLAGS.characteristics)

if FLAGS.characteristics == 'lid':
    train_characteristics_file = os.path.join(characteristics_dir, 'k_{}_batch_{}_train_noisy_{}.npy'.format(FLAGS.k_nearest, 100, FLAGS.with_noise))
    test_characteristics_file  = os.path.join(characteristics_dir, 'k_{}_batch_{}_test_noisy_{}.npy'.format(FLAGS.k_nearest, 100, FLAGS.with_noise))
elif FLAGS.characteristics == 'mahalanobis':
    train_characteristics_file = os.path.join(characteristics_dir, 'magnitude_{}_scale_{}_train_noisy_{}.npy'.format(FLAGS.magnitude, FLAGS.rgb_scale, FLAGS.with_noise))
    test_characteristics_file  = os.path.join(characteristics_dir, 'magnitude_{}_scale_{}_test_noisy_{}.npy'.format(FLAGS.magnitude, FLAGS.rgb_scale, FLAGS.with_noise))
elif FLAGS.characteristics == 'nnif':
    train_characteristics_file = os.path.join(characteristics_dir, 'max_indices_{}_train_ablation_{}_noisy_{}.npy'.format(FLAGS.max_indices, FLAGS.ablation, FLAGS.with_noise))
    test_characteristics_file  = os.path.join(characteristics_dir, 'max_indices_{}_test_ablation_{}_noisy_{}.npy'.format(FLAGS.max_indices, FLAGS.ablation, FLAGS.with_noise))
elif FLAGS.characteristics == 'dknn':
    train_characteristics_file = os.path.join(characteristics_dir, 'k_{}_train_noisy_{}.npy'.format(FLAGS.k_nearest, FLAGS.with_noise))
    test_characteristics_file  = os.path.join(characteristics_dir, 'k_{}_test_noisy_{}.npy'.format(FLAGS.k_nearest, FLAGS.with_noise))
else:
    raise AssertionError('{} is not supported'.format(FLAGS.characteristics))

def load_characteristics(characteristics_file):
    X, Y = None, None
    data = np.load(characteristics_file)
    if X is None:
        X = data[:, :-1]
    if Y is None:
        Y = data[:, -1]  # labels only need to load once

    return X, Y


print("Loading train attack: %s" % FLAGS.attack)
# X, Y = load_characteristics(characteristics_file)
X_train, Y_train = load_characteristics(train_characteristics_file)
X_test, Y_test   = load_characteristics(test_characteristics_file)

scaler  = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# test attack is the same as training attack
# X_train, Y_train, X_test, Y_test = block_split(X, Y)

print("Train data size: ", X_train.shape)
print("Test data size: ", X_test.shape)

## Build detector
print("LR Detector on [dataset: %s, test_attack: %s, characteristics: %s, ablation: %s]:" % (FLAGS.dataset, FLAGS.attack, FLAGS.characteristics, FLAGS.ablation))
lr = train_lr(X_train, Y_train)

## Evaluate detector
y_pred       = lr.predict_proba(X_test)[:, 1]
y_label_pred = lr.predict(X_test)

# AUC
_, _, auc_score = compute_roc(Y_test, y_pred, plot=True)
precision = precision_score(Y_test, y_label_pred)
recall    = recall_score(Y_test, y_label_pred)
acc       = accuracy_score(Y_test, y_label_pred)
print('Detector ROC-AUC score: {}, accuracy: {}, precision: {}, recall: {}'.format(auc_score, acc, precision, recall))