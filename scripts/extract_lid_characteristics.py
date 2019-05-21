from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
import platform
# Force matplotlib to not use any Xwindows backend.
if platform.system() == 'Linux':
    matplotlib.use('Agg')

import logging
import numpy as np
import tensorflow as tf
import os
import imageio

import darkon.darkon as darkon

from cleverhans.attacks import FastGradientMethod, DeepFool, SaliencyMapMethod, CarliniWagnerL2
from tensorflow.python.platform import flags
from cleverhans.loss import CrossEntropy, WeightDecay, WeightedSum
from tensorflow_TB.lib.models.darkon_replica_model import DarkonReplica
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from tensorflow_TB.utils.misc import one_hot
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tensorflow_TB.lib.datasets.influence_feeder_val_test import MyFeederValTest
from tensorflow_TB.utils.misc import np_evaluate
import pickle
from cleverhans.utils import random_targets
from cleverhans.evaluation import batch_eval

STDEVS = {
    'cifar10': {'deepfool': 0.007, 'cw': 0.007}
}

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_string('dataset', 'cifar10', 'dataset: cifar10/100 or svhn')
flags.DEFINE_string('attack', 'deepfool', 'adversarial attack: deepfool, jsma, cw')
flags.DEFINE_bool('targeted', False, 'whether or not the adversarial attack is targeted')

if FLAGS.dataset == 'cifar10':
    _classes = (
        'airplane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    )
    ARCH_NAME = 'model1'
    CHECKPOINT_NAME = 'cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000'
    LABEL_SMOOTHING = 0.1
elif FLAGS.dataset == 'cifar100':
    _classes = (
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    )
    ARCH_NAME = 'model_cifar_100'
    CHECKPOINT_NAME = 'cifar100/log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01'
    LABEL_SMOOTHING = 0.01
elif FLAGS.dataset == 'svhn':
    _classes = (
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    )
    ARCH_NAME = 'model_svhn'
    CHECKPOINT_NAME = 'svhn/log_120519_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1257'
    LABEL_SMOOTHING = 0.1
else:
    raise AssertionError('dataset {} not supported'.format(FLAGS.dataset))

# Set TF random seed to improve reproducibility
superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# get records from training
model_dir     = os.path.join('/data/gilad/logs/influence', CHECKPOINT_NAME)
attack_dir    = os.path.join(model_dir, FLAGS.attack)
if FLAGS.targeted:
    attack_dir = attack_dir + '_targeted'

print('loading train mini indices from {}'.format(os.path.join(model_dir, 'train_mini_indices.npy')))
mini_train_inds = np.load(os.path.join(model_dir, 'train_mini_indices.npy'))
val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))
feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                         test_val_set=False, mini_train_inds=mini_train_inds)

# get the dataset
X_train     , y_train      = feeder.train_data     , feeder.train_label        # real train set (49k):
X_train_mini, y_train_mini = feeder.mini_train_data, feeder.mini_train_label   # mini train set (just 5k)
X_val       , y_val        = feeder.val_data       , feeder.val_label          # val set (1k)
X_test      , y_test       = feeder.test_data      , feeder.test_label         # test set
y_train_sparse             = y_train.argmax(axis=-1).astype(np.int32)
y_train_mini_sparse        = y_train_mini.argmax(axis=-1).astype(np.int32)
y_val_sparse               = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse              = y_test.argmax(axis=-1).astype(np.int32)

# if the attack is targeted, fetch the targets
if FLAGS.targeted:
    # get also the adversarial labels of the val and test sets
    if not os.path.isfile(os.path.join(attack_dir, 'y_val_targets.npy')):
        y_val_targets  = random_targets(y_val_sparse , feeder.num_classes)
        y_test_targets = random_targets(y_test_sparse, feeder.num_classes)
        assert (y_val_targets.argmax(axis=1)  != y_val_sparse).all()
        assert (y_test_targets.argmax(axis=1) != y_test_sparse).all()
        np.save(os.path.join(attack_dir, 'y_val_targets.npy') , y_val_targets)
        np.save(os.path.join(attack_dir, 'y_test_targets.npy'), y_test_targets)
    else:
        y_val_targets  = np.load(os.path.join(attack_dir, 'y_val_targets.npy'))
        y_test_targets = np.load(os.path.join(attack_dir, 'y_test_targets.npy'))

# fetch the predictions and embedding vectors
x_train_preds         = np.load(os.path.join(model_dir, 'x_train_preds.npy'))
x_train_features      = np.load(os.path.join(model_dir, 'x_train_features.npy'))

x_train_mini_preds    = np.load(os.path.join(model_dir, 'x_train_mini_preds.npy'))
x_train_mini_features = np.load(os.path.join(model_dir, 'x_train_mini_features.npy'))

x_val_preds           = np.load(os.path.join(model_dir, 'x_val_preds.npy'))
x_val_features        = np.load(os.path.join(model_dir, 'x_val_features.npy'))

x_test_preds          = np.load(os.path.join(model_dir, 'x_test_preds.npy'))
x_test_features       = np.load(os.path.join(model_dir, 'x_test_features.npy'))

X_val_adv             = np.load(os.path.join(attack_dir, 'X_val_adv.npy'))
x_val_preds_adv       = np.load(os.path.join(attack_dir, 'x_val_preds_adv.npy'))
x_val_features_adv    = np.load(os.path.join(attack_dir, 'x_val_features_adv.npy'))

X_test_adv            = np.load(os.path.join(attack_dir, 'X_test_adv.npy'))
x_test_preds_adv      = np.load(os.path.join(attack_dir, 'x_test_preds_adv.npy'))
x_test_features_adv   = np.load(os.path.join(attack_dir, 'x_test_features_adv.npy'))

# quick computations of accuracies
train_acc    = np.mean(y_train_sparse == x_train_preds)
val_acc      = np.mean(y_val_sparse   == x_val_preds)
test_acc     = np.mean(y_test_sparse  == x_test_preds)
val_adv_acc  = np.mean(y_val_sparse   == x_val_preds_adv)
test_adv_acc = np.mean(y_test_sparse  == x_test_preds_adv)
print('train set acc: {}\nvalidation set acc: {}\ntest set acc: {}'.format(train_acc, val_acc, test_acc))
print('adversarial ({}) validation set acc: {}\nadversarial ({}) test set acc: {}'.format(FLAGS.attack, val_adv_acc, FLAGS.attack, test_adv_acc))

# what are the indices of the cifar10 set which the network succeeded classifying correctly,
# but the adversarial attack changed to a different class?
info = {}
info['val'] = {}
for i, set_ind in enumerate(feeder.val_inds):
    info['val'][i] = {}
    net_succ    = x_val_preds[i] == y_val_sparse[i]
    attack_succ = x_val_preds[i] != x_val_preds_adv[i]
    info['val'][i]['global_index'] = set_ind
    info['val'][i]['net_succ']     = net_succ
    info['val'][i]['attack_succ']  = attack_succ
info['test'] = {}
for i, set_ind in enumerate(feeder.test_inds):
    info['test'][i] = {}
    net_succ    = x_test_preds[i] == y_test_sparse[i]
    attack_succ = x_test_preds[i] != x_test_preds_adv[i]
    info['test'][i]['global_index'] = set_ind
    info['test'][i]['net_succ']     = net_succ
    info['test'][i]['attack_succ']  = attack_succ

info_file = os.path.join(attack_dir, 'info.pkl')
print('loading info as pickle from {}'.format(info_file))
with open(info_file, 'rb') as handle:
    info_old = pickle.load(handle)
assert info == info_old

# get model and placeholders
superseed = 15101985
rand_gen = np.random.RandomState(superseed)
tf.set_random_seed(superseed)
config_args = dict(allow_soft_placement=True)
sess = tf.Session(config=tf.ConfigProto(**config_args))

img_rows, img_cols, nchannels = X_test.shape[1:4]
nb_classes = y_test.shape[1]
x     = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels), name='x')
y     = tf.placeholder(tf.float32, shape=(None, nb_classes), name='y')

eval_params = {'batch_size': FLAGS.batch_size}
model = DarkonReplica(scope=ARCH_NAME, nb_classes=feeder.num_classes, n=5, input_shape=[32, 32, 3])
preds      = model.get_predicted_class(x)
logits     = model.get_logits(x)
embeddings = model.get_embeddings(x)
saver = tf.train.Saver()
checkpoint_path = os.path.join(model_dir, 'best_model.ckpt')
saver.restore(sess, checkpoint_path)

# get noisy images
def get_noisy_samples(X_test, X_test_adv, std=STDEVS[FLAGS.dataset][FLAGS.attack]):
    """ Add Gaussian noise to the samples """
    X_test_noisy = np.clip(X_test + rand_gen.normal(loc=0.0, scale=std, size=X_test.shape), 0, 1)
    return X_test_noisy

noisy_file = os.path.join(attack_dir, 'X_test_noisy.npy')

# DEBUG: testing different scale so that L2 perturbation is the same
for std in np.arange(0.001, 0.03, 0.0005):
    X_test_noisy = get_noisy_samples(X_test, X_test_adv, std)

    for s_type, subset in zip(['noisy'], [X_test_noisy]):
        acc = model_eval(sess, x, y, logits, subset, y_test, args=eval_params)
        print("Noise %0.5f: Model accuracy on the %s test set: %0.2f%%" % (std, s_type, 100 * acc))
        # Compute and display average perturbation sizes
        if not s_type == 'normal':
            diff    = subset.reshape((len(X_test), -1)) - X_test.reshape((len(X_test), -1))
            l2_diff = np.linalg.norm(diff, axis=1).mean()
            print("Noise %0.5f: Average L-2 perturbation size of the %s test set: %0.2f" % (std, s_type, l2_diff))

# for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_test, X_test_noisy, X_test_adv]):
#     acc = model_eval(sess, x, y, logits, subset, y_test, args=eval_params)
#     print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc))
#     # Compute and display average perturbation sizes
#     if not s_type == 'normal':
#         diff    = subset.reshape((len(X_test), -1)) - X_test.reshape((len(X_test), -1))
#         l2_diff = np.linalg.norm(diff, axis=1).mean()
#         print("Average L-2 perturbation size of the %s test set: %0.2f" % (s_type, l2_diff))


