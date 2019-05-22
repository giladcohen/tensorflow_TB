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
flags.DEFINE_string('attack', 'deepfool', 'adversarial attack: deepfool, jsma, cw')
flags.DEFINE_bool('targeted', False, 'whether or not the adversarial attack is targeted')
flags.DEFINE_string('characteristics', 'nnif', 'type of defence')
flags.DEFINE_integer('k_nearest', 100, 'number of nearest neighbors to use for LID detection')
flags.DEFINE_float('magnitude', 0.002, 'magnitude for mahalanobis detection')
flags.DEFINE_integer('max_indices', 800, 'maximum number of helpful indices to use in NNIF detection')


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
    characteristics_file = os.path.join(characteristics_dir, 'k_{}_batch_{}.npy'.format(FLAGS.k_nearest, 100))
elif FLAGS.characteristics == 'mahalanobis':
    characteristics_file = os.path.join(characteristics_dir, 'magnitude_{}.npy'.format(FLAGS.magnitude))
elif FLAGS.characteristics == 'nnif':
    characteristics_file = os.path.join(characteristics_dir, 'max_indices_{}.npy'.format(FLAGS.max_indices))
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
X, Y = load_characteristics(characteristics_file)

scaler = MinMaxScaler().fit(X)

# test attack is the same as training attack
X_train, Y_train, X_test, Y_test = block_split(X, Y)

print("Train data size: ", X_train.shape)
print("Test data size: ", X_test.shape)

## Build detector
print("LR Detector on [dataset: %s, test_attack: %s, characteristics: %s]:" % (FLAGS.dataset, FLAGS.attack, FLAGS.characteristics))
lr = train_lr(X_train, Y_train)

## Evaluate detector
y_pred       = lr.predict_proba(X_test)[:, 1]
y_label_pred = lr.predict(X_test)

# AUC
_, _, auc_score = compute_roc(Y_test, y_pred, plot=True)
precision = precision_score(Y_test, y_label_pred)
recall    = recall_score(Y_test, y_label_pred)
acc       = accuracy_score(Y_test, y_label_pred)
print('Detector ROC-AUC score: %0.4f, accuracy: %.4f, precision: %.4f, recall: %.4f' % (auc_score, acc, precision, recall))