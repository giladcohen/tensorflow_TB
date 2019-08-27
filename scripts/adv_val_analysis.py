from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import imageio
from sklearn.neighbors import NearestNeighbors

from tensorflow.python.platform import flags
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.fallback_to_cm'] = True
import matplotlib.pyplot as plt
from tensorflow_TB.lib.datasets.influence_feeder_val_test import MyFeederValTest
import pickle

STDEVS = {
    'val' : {'cifar10': {'deepfool': 0.00861, 'cw': 0.003081}},
    'test': {'cifar10': {'deepfool': 0.00796, 'cw': 0.003057}}
}

FLAGS = flags.FLAGS

flags.DEFINE_string('set', 'val', 'val or test set to evaluate')
flags.DEFINE_bool('use_train_mini', False, 'Whether or not to use 5000 training samples instead of 49000')
flags.DEFINE_string('dataset', 'cifar10', 'datasset: cifar10/100 or svhn')
flags.DEFINE_string('attack', 'deepfool', 'adversarial attack: deepfool, jsma, cw')
flags.DEFINE_string('characteristics', 'nnif', 'type of defence')
flags.DEFINE_integer('max_indices', 800, 'maximum number of helpful indices to use in NNIF detection')
flags.DEFINE_string('analysis', 'original_knn_dist', 'analysis type: features/original_knn_dist')
flags.DEFINE_bool('refine_val', False, 'Considering only correct predictions for the validation set')
flags.DEFINE_bool('refine_test', False, 'Considering only correct predictions for the validation set')

flags.DEFINE_string('mode', 'null', 'to bypass pycharm bug')
flags.DEFINE_string('port', 'null', 'to bypass pycharm bug')


if FLAGS.set == 'val':
    test_val_set = True
    WORKSPACE = 'influence_workspace_validation'
    USE_TRAIN_MINI = False
else:
    test_val_set = False
    WORKSPACE = 'influence_workspace_test_mini'
    USE_TRAIN_MINI = True

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

TARGETED = FLAGS.attack != 'deepfool'
superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# get records from training
model_dir  = os.path.join('/data/gilad/logs/influence', CHECKPOINT_NAME)
attack_dir = os.path.join(model_dir, FLAGS.attack)
if TARGETED:
    attack_dir = attack_dir + '_targeted'
characteristics_dir = os.path.join(attack_dir, FLAGS.characteristics)

mini_train_inds = None
if USE_TRAIN_MINI:
    print('loading train mini indices from {}'.format(os.path.join(model_dir, 'train_mini_indices.npy')))
    mini_train_inds = np.load(os.path.join(model_dir, 'train_mini_indices.npy'))

val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))
feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                         test_val_set=test_val_set, mini_train_inds=mini_train_inds)


# get the dataset
X_train     , y_train      = feeder.train_data     , feeder.train_label        # real train set (49k):
X_val       , y_val        = feeder.val_data       , feeder.val_label          # val set (1k)
X_test      , y_test       = feeder.test_data      , feeder.test_label         # test set
y_train_sparse             = y_train.argmax(axis=-1).astype(np.int32)
y_val_sparse               = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse              = y_test.argmax(axis=-1).astype(np.int32)

if USE_TRAIN_MINI:
    X_train_mini, y_train_mini = feeder.mini_train_data, feeder.mini_train_label   # mini train set (just 5k)
    y_train_mini_sparse        = y_train_mini.argmax(axis=-1).astype(np.int32)

# if the attack is targeted, fetch the targets
if TARGETED:
    y_val_targets  = np.load(os.path.join(attack_dir, 'y_val_targets.npy'))
    y_test_targets = np.load(os.path.join(attack_dir, 'y_test_targets.npy'))

# fetch the predictions and embedding vectors
x_train_preds         = np.load(os.path.join(model_dir, 'x_train_preds.npy'))
x_train_features      = np.load(os.path.join(model_dir, 'x_train_features.npy'))

if USE_TRAIN_MINI:
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

def get_noisy_samples(X, std):
    """ Add Gaussian noise to the samples """
    # std = STDEVS[subset][FLAGS.dataset][FLAGS.attack]
    X_noisy = np.clip(X + rand_gen.normal(loc=0.0, scale=std, size=X.shape), 0, 1)
    return X_noisy

noisy_file = os.path.join(attack_dir, 'X_val_noisy.npy')
if os.path.isfile(noisy_file):
    print('Loading {} val noisy samples from {}'.format(FLAGS.dataset, noisy_file))
    X_val_noisy = np.load(noisy_file)
else:
    print('WARNING: file {} does not exist'.format(noisy_file))

noisy_file = os.path.join(attack_dir, 'X_test_noisy.npy')
if os.path.isfile(noisy_file):
    print('Loading {} noisy samples from {}'.format(FLAGS.dataset, noisy_file))
    X_test_noisy = np.load(noisy_file)
else:
    print('WARNING: file {} does not exist'.format(noisy_file))

# print stats for val
# for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_val, X_val_noisy, X_val_adv]):
#     # acc = model_eval(sess, x, y, logits, subset, y_val, args=eval_params)
#     # print("Model accuracy on the %s val set: %0.2f%%" % (s_type, 100 * acc))
#     # Compute and display average perturbation sizes
#     if not s_type == 'normal':
#         # print for test:
#         diff    = subset.reshape((len(subset), -1)) - X_val.reshape((len(subset), -1))
#         l2_diff = np.linalg.norm(diff, axis=1).mean()
#         print("Average L-2 perturbation size of the %s val set: %0.4f" % (s_type, l2_diff))
#
# # print stats for test
# for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_test, X_test_noisy, X_test_adv]):
#     # acc = model_eval(sess, x, y, logits, subset, y_test, args=eval_params)
#     # print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc))
#     # Compute and display average perturbation sizes
#     if not s_type == 'normal':
#         # print for test:
#         diff    = subset.reshape((len(subset), -1)) - X_test.reshape((len(subset), -1))
#         l2_diff = np.linalg.norm(diff, axis=1).mean()
#         print("Average L-2 perturbation size of the %s test set: %0.4f" % (s_type, l2_diff))

if FLAGS.refine_val:
    val_inds_correct  = np.where(x_val_preds == y_val_sparse)[0]
    print("Number of correctly val predict images: %s" % (len(val_inds_correct)))
    X_val              = X_val[val_inds_correct]
    # X_val_noisy        = X_val_noisy[val_inds_correct]
    X_val_adv          = X_val_adv[val_inds_correct]
    x_val_preds        = x_val_preds[val_inds_correct]
    x_val_features     = x_val_features[val_inds_correct]
    x_val_preds_adv    = x_val_preds_adv[val_inds_correct]
    x_val_features_adv = x_val_features_adv[val_inds_correct]
    y_val              = y_val[val_inds_correct]
    y_val_sparse       = y_val_sparse[val_inds_correct]

if FLAGS.refine_test:
    test_inds_correct = np.where(x_test_preds == y_test_sparse)[0]
    print("Number of correctly test predict images: %s" % (len(test_inds_correct)))
    X_test              = X_test[test_inds_correct]
    # X_test_noisy        = X_test_noisy[test_inds_correct]
    X_test_adv          = X_test_adv[test_inds_correct]
    x_test_preds        = x_test_preds[test_inds_correct]
    x_test_features     = x_test_features[test_inds_correct]
    x_test_preds_adv    = x_test_preds_adv[test_inds_correct]
    x_test_features_adv = x_test_features_adv[test_inds_correct]
    y_test              = y_test[test_inds_correct]
    y_test_sparse       = y_test_sparse[test_inds_correct]

print("X_val: "       , X_val.shape)
# print("X_val_noisy: " , X_val_noisy.shape)
print("X_val_adv: "   , X_val_adv.shape)

print("X_test: "      , X_test.shape)
# print("X_test_noisy: ", X_test_noisy.shape)
print("X_test_adv: "  , X_test_adv.shape)

sub_relevant_indices = [ind for ind in info[FLAGS.set]]
relevant_indices     = [info[FLAGS.set][ind]['global_index'] for ind in sub_relevant_indices]

if FLAGS.analysis == 'features':
    real = {}
    pred = {}
    pred_correct = {}
    pred_incorrect = {}
    adv = {}
    adv_succ = {}
    adv_fail = {}

    for i in range(1000):  # 1000 = helpful_ranks.shape[0]
        real[i]           = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
        pred[i]           = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
        pred_correct[i]   = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
        pred_incorrect[i] = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
        adv[i]            = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
        adv_succ[i]       = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
        adv_fail[i]       = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}

    for i, sub_index in enumerate(sub_relevant_indices):
        global_index = feeder.val_inds[sub_index]
        print('start analyzing on sample i={}. global_index={}'.format(i, global_index))
        assert global_index == relevant_indices[i]

        real_label = y_val_sparse[sub_index]
        pred_label = x_val_preds[sub_index]
        adv_label  = x_val_preds_adv[sub_index]

        net_succ    = info[FLAGS.set][sub_index]['net_succ']
        attack_succ = info[FLAGS.set][sub_index]['attack_succ']

        if attack_succ:
            assert pred_label != adv_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)
        if net_succ:
            assert pred_label == real_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)

        index_dir = os.path.join(model_dir, 'val', 'val_index_{}'.format(global_index))

        # collect real
        helpful_ranks = np.load(os.path.join(index_dir, 'real', 'helpful_ranks.npy'))
        helpful_dists = np.load(os.path.join(index_dir, 'real', 'helpful_dists.npy'))
        harmful_ranks = np.load(os.path.join(index_dir, 'real', 'harmful_ranks.npy'))
        harmful_dists = np.load(os.path.join(index_dir, 'real', 'harmful_dists.npy'))

        for k in range(helpful_ranks.shape[0]):
            real[k]['+rank'].append(helpful_ranks[k])
            real[k]['+dist'].append(helpful_dists[k])
            real[k]['-rank'].append(harmful_ranks[k])
            real[k]['-dist'].append(harmful_dists[k])

        # collect pred
        if net_succ:
            # same as real
            for k in range(helpful_ranks.shape[0]):
                pred[k]['+rank'].append(helpful_ranks[k])
                pred[k]['+dist'].append(helpful_dists[k])
                pred[k]['-rank'].append(harmful_ranks[k])
                pred[k]['-dist'].append(harmful_dists[k])
                pred_correct[k]['+rank'].append(helpful_ranks[k])
                pred_correct[k]['+dist'].append(helpful_dists[k])
                pred_correct[k]['-rank'].append(harmful_ranks[k])
                pred_correct[k]['-dist'].append(harmful_dists[k])
        else:
            # has its own folder - collect pred
            helpful_ranks = np.load(os.path.join(index_dir, 'pred', 'helpful_ranks.npy'))
            helpful_dists = np.load(os.path.join(index_dir, 'pred', 'helpful_dists.npy'))
            harmful_ranks = np.load(os.path.join(index_dir, 'pred', 'harmful_ranks.npy'))
            harmful_dists = np.load(os.path.join(index_dir, 'pred', 'harmful_dists.npy'))
            for k in range(helpful_ranks.shape[0]):
                pred[k]['+rank'].append(helpful_ranks[k])
                pred[k]['+dist'].append(helpful_dists[k])
                pred[k]['-rank'].append(harmful_ranks[k])
                pred[k]['-dist'].append(harmful_dists[k])
                pred_incorrect[k]['+rank'].append(helpful_ranks[k])
                pred_incorrect[k]['+dist'].append(helpful_dists[k])
                pred_incorrect[k]['-rank'].append(harmful_ranks[k])
                pred_incorrect[k]['-dist'].append(harmful_dists[k])

        # collect adv
        helpful_ranks = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'helpful_ranks.npy'))
        helpful_dists = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'helpful_dists.npy'))
        harmful_ranks = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'harmful_ranks.npy'))
        harmful_dists = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'harmful_dists.npy'))
        adv[k]['+rank'].append(helpful_ranks[k])
        adv[k]['+dist'].append(helpful_dists[k])
        adv[k]['-rank'].append(harmful_ranks[k])
        adv[k]['-dist'].append(harmful_dists[k])
        if attack_succ:
            for k in range(helpful_ranks.shape[0]):
                adv_succ[k]['+rank'].append(helpful_ranks[k])
                adv_succ[k]['+dist'].append(helpful_dists[k])
                adv_succ[k]['-rank'].append(harmful_ranks[k])
                adv_succ[k]['-dist'].append(harmful_dists[k])
        else:
            for k in range(helpful_ranks.shape[0]):
                adv_fail[k]['+rank'].append(helpful_ranks[k])
                adv_fail[k]['+dist'].append(helpful_dists[k])
                adv_fail[k]['-rank'].append(harmful_ranks[k])
                adv_fail[k]['-dist'].append(harmful_dists[k])

    if FLAGS.attack == 'deepfool':
        plt.close()
        plt.rcParams['interactive'] = False
        subpos = np.array([0.35, 0.25, 0.5, 0.4])
        fig = plt.figure(figsize=(8.0, 8.0))

        # +rank
        ax1 = fig.add_subplot(311)
        all_real = []
        all_pred = []
        all_pred_correct = []
        all_pred_incorrect = []
        all_adv = []
        all_adv_succ = []
        all_adv_fail = []
        for i in range(FLAGS.max_indices):
            all_real.extend(real[i]['+rank'])
            all_pred.extend(pred[i]['+rank'])
            all_pred_correct.extend(pred_correct[i]['+rank'])
            all_pred_incorrect.extend(pred_incorrect[i]['+rank'])
            all_adv.extend(adv[i]['+rank'])
            all_adv_succ.extend(adv_succ[i]['+rank'])
            all_adv_fail.extend(adv_fail[i]['+rank'])

        all_values = all_real + all_pred + all_adv
        rangee = (np.min(all_values), np.max(all_values))

        ax1.hist(all_real, range=rangee, label='real', alpha=0.25, bins=300, density=True)
        ax1.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=300, density=True)
        ax1.set_title('Deepfool')
        ax1.legend(loc='upper right')
        ax1.set_ylabel('Helpful ranks')
        # ax1.set_xlabel('+rank', horizontalalignment='right')
        ax1.set_xlim(0, 10000)

        # + dist
        ax2 = fig.add_subplot(312)
        all_real = []
        all_pred = []
        all_pred_correct = []
        all_pred_incorrect = []
        all_adv = []
        all_adv_succ = []
        all_adv_fail = []
        for i in range(FLAGS.max_indices):
            all_real.extend(real[i]['+dist'])
            all_pred.extend(pred[i]['+dist'])
            all_pred_correct.extend(pred_correct[i]['+dist'])
            all_pred_incorrect.extend(pred_incorrect[i]['+dist'])
            all_adv.extend(adv[i]['+dist'])
            all_adv_succ.extend(adv_succ[i]['+dist'])
            all_adv_fail.extend(adv_fail[i]['+dist'])

        all_values = all_real + all_pred + all_adv
        rangee = (np.min(all_values), np.max(all_values))

        ax2.hist(all_real, range=rangee, label='real', alpha=0.25, bins=300, density=True)
        ax2.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=300, density=True)
        ax2.hist(all_pred_incorrect, range=rangee, label='pred(incorrect)', alpha=0.25, bins=200, density=True)
        ax2.legend(loc='upper center')
        ax2.set_ylabel('Helpful distances')
        # ax2.set_xlabel('+dist', horizontalalignment='right')
        ax2.set_xlim(0, 8)

        # - dist
        ax3 = fig.add_subplot(313)
        all_real = []
        all_pred = []
        all_pred_correct = []
        all_pred_incorrect = []
        all_adv = []
        all_adv_succ = []
        all_adv_fail = []
        for i in range(FLAGS.max_indices):
            all_real.extend(real[i]['-dist'])
            all_pred.extend(pred[i]['-dist'])
            all_pred_correct.extend(pred_correct[i]['-dist'])
            all_pred_incorrect.extend(pred_incorrect[i]['-dist'])
            all_adv.extend(adv[i]['-dist'])
            all_adv_succ.extend(adv_succ[i]['-dist'])
            all_adv_fail.extend(adv_fail[i]['-dist'])

        all_values = all_real + all_pred + all_adv
        rangee = (np.min(all_values), np.max(all_values))

        ax3.hist(all_real, range=rangee, label='real', alpha=0.25, bins=300, density=True)
        ax3.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=300, density=True)
        ax3.legend(loc='upper right')
        ax3.set_ylabel('Harmful distances')
        # ax2.set_xlabel('+dist', horizontalalignment='right')
        ax3.set_xlim(0, 8)

        plt.savefig('deepfool_rank_and_dist_hists.png', dpi=350)

    elif FLAGS.attack == 'cw':
        plt.close()
        plt.rcParams['interactive'] = False
        subpos = np.array([0.35, 0.25, 0.5, 0.4])
        fig = plt.figure(figsize=(8.0, 8.0))

        # +rank
        ax1 = fig.add_subplot(311)
        all_real = []
        all_pred = []
        all_pred_correct = []
        all_pred_incorrect = []
        all_adv = []
        all_adv_succ = []
        all_adv_fail = []
        for i in range(FLAGS.max_indices):
            all_real.extend(real[i]['+rank'])
            all_pred.extend(pred[i]['+rank'])
            all_pred_correct.extend(pred_correct[i]['+rank'])
            all_pred_incorrect.extend(pred_incorrect[i]['+rank'])
            all_adv.extend(adv[i]['+rank'])
            all_adv_succ.extend(adv_succ[i]['+rank'])
            all_adv_fail.extend(adv_fail[i]['+rank'])

        all_values = all_real + all_pred + all_adv
        rangee = (np.min(all_values), np.max(all_values))

        ax1.hist(all_real, range=rangee, label='real', alpha=0.25, bins=300, density=True)
        ax1.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=300, density=True)
        ax1.set_title('Carlini-Wagner')
        ax1.legend(loc='upper right')
        # ax1.set_ylabel(r'$\mathbb{R}^{\Uparrow}$ PDF')
        ax1.set_ylabel('Helpful ranks')
        # ax1.set_xlabel('+rank', horizontalalignment='right')
        ax1.set_xlim(0, 10000)

        # + dist
        ax2 = fig.add_subplot(312)
        all_real = []
        all_pred = []
        all_pred_correct = []
        all_pred_incorrect = []
        all_adv = []
        all_adv_succ = []
        all_adv_fail = []
        for i in range(FLAGS.max_indices):
            all_real.extend(real[i]['+dist'])
            all_pred.extend(pred[i]['+dist'])
            all_pred_correct.extend(pred_correct[i]['+dist'])
            all_pred_incorrect.extend(pred_incorrect[i]['+dist'])
            all_adv.extend(adv[i]['+dist'])
            all_adv_succ.extend(adv_succ[i]['+dist'])
            all_adv_fail.extend(adv_fail[i]['+dist'])

        all_values = all_real + all_pred + all_adv
        rangee = (np.min(all_values), np.max(all_values))

        ax2.hist(all_real, range=rangee, label='real', alpha=0.25, bins=300, density=True)
        ax2.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=300, density=True)
        ax2.hist(all_pred_incorrect, range=rangee, label='pred(incorrect)', alpha=0.25, bins=200, density=True)
        ax2.legend(loc='upper right')
        ax2.set_ylabel('Helpful distances')
        # ax2.set_xlabel('+dist', horizontalalignment='right')
        ax2.set_xlim(0, 8)

        # - dist
        ax3 = fig.add_subplot(313)
        all_real = []
        all_pred = []
        all_pred_correct = []
        all_pred_incorrect = []
        all_adv = []
        all_adv_succ = []
        all_adv_fail = []
        for i in range(FLAGS.max_indices):
            all_real.extend(real[i]['-dist'])
            all_pred.extend(pred[i]['-dist'])
            all_pred_correct.extend(pred_correct[i]['-dist'])
            all_pred_incorrect.extend(pred_incorrect[i]['-dist'])
            all_adv.extend(adv[i]['-dist'])
            all_adv_succ.extend(adv_succ[i]['-dist'])
            all_adv_fail.extend(adv_fail[i]['-dist'])

        all_values = all_real + all_pred + all_adv
        rangee = (np.min(all_values), np.max(all_values))

        ax3.hist(all_real, range=rangee, label='real', alpha=0.25, bins=300, density=True)
        ax3.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=300, density=True)
        ax3.legend(loc='upper right')
        ax3.set_ylabel('Harmful distances')
        # ax2.set_xlabel('+dist', horizontalalignment='right')
        ax3.set_xlim(0, 8)

        plt.savefig('carlini_wagner_rank_and_dist_hists.png', dpi=350)

elif FLAGS.analysis == 'original_knn_dist':
    # histogram range dictionary
    range_dict = {'cifar10': {'deepfool': (0, 10)}}

    # build the knn model
    knn = NearestNeighbors(n_neighbors=feeder.get_train_size(), p=2, n_jobs=20, algorithm='brute')
    knn.fit(x_train_features)
    if test_val_set:
        print('predicting knn for all val set')
        features = x_val_features
        features_adv = x_val_features_adv
    else:
        print('predicting knn for all test set')
        features = x_test_features
        features_adv = x_test_features_adv
    print('predicting knn dist/indices for normal image')
    all_neighbor_dists    , all_neighbor_indices     = knn.kneighbors(features, return_distance=True)
    print('predicting knn dist/indices for adv image')
    all_neighbor_dists_adv, all_neighbor_indices_adv = knn.kneighbors(features_adv, return_distance=True)

    # initializing
    real = {}
    pred = {}
    pred_correct = {}
    pred_incorrect = {}
    adv = {}
    adv_succ = {}
    adv_fail = {}

    k = 50  # number of nearest neighbors to consider
    for ki in range(k):
        pred[ki]           = {'dist': []}
        pred_correct[ki]   = {'dist': []}
        pred_incorrect[ki] = {'dist': []}
        adv[ki]            = {'dist': []}
        adv_succ[ki]       = {'dist': []}
        adv_fail[ki]       = {'dist': []}

    # initializing global stats
    num_pred = len(sub_relevant_indices)
    num_pred_correct   = len([ind for ind in info[FLAGS.set] if info[FLAGS.set][ind]['net_succ']])
    num_pred_incorrect = len([ind for ind in info[FLAGS.set] if not info[FLAGS.set][ind]['net_succ']])
    num_adv = len(sub_relevant_indices)
    num_adv_succ       = len([ind for ind in info[FLAGS.set] if info[FLAGS.set][ind]['attack_succ']])
    num_adv_fail       = len([ind for ind in info[FLAGS.set] if not info[FLAGS.set][ind]['attack_succ']])

    glb_pred = {'mean_dist': [], 'median_dist': [], 'max_dist': [], 'min_dist': []}
    glb_pred_correct = {'mean_dist': [], 'median_dist': [], 'max_dist': [], 'min_dist': []}
    glb_pred_incorrect = {'mean_dist': [], 'median_dist': [], 'max_dist': [], 'min_dist': []}
    glb_adv = {'mean_dist': [], 'median_dist': [], 'max_dist': [], 'min_dist': [], 'mean_dist_ratio': []}
    glb_adv_succ = {'mean_dist': [], 'median_dist': [], 'max_dist': [], 'min_dist': [], 'mean_dist_ratio': []}
    glb_adv_fail = {'mean_dist': [], 'median_dist': [], 'max_dist': [], 'min_dist': [], 'mean_dist_ratio': []}

    assert (np.array(sub_relevant_indices) == np.arange(features.shape[0])).all(), "sub_relevant_indices must be continuous"

    # collecting stats
    for i, sub_index in enumerate(sub_relevant_indices):
        global_index = feeder.val_inds[sub_index]
        print('start analyzing on sample i={}. global_index={}'.format(i, global_index))
        assert global_index == relevant_indices[i]
        net_succ    = info[FLAGS.set][sub_index]['net_succ']
        attack_succ = info[FLAGS.set][sub_index]['attack_succ']

        # fill local
        for ki in range(k):
            pred[ki]['dist'].append(all_neighbor_dists[i, ki])
            if net_succ:
                pred_correct[ki]['dist'].append(all_neighbor_dists[i, ki])
            else:
                pred_incorrect[ki]['dist'].append(all_neighbor_dists[i, ki])

        # fill global
        mean   = np.mean(all_neighbor_dists[i, :k])
        median = np.median(all_neighbor_dists[i, :k])
        max    = np.max(all_neighbor_dists[i, :k])
        min    = np.min(all_neighbor_dists[i, :k])

        glb_pred['mean_dist'].append(mean)
        glb_pred['median_dist'].append(median)
        glb_pred['max_dist'].append(max)
        glb_pred['min_dist'].append(min)
        if net_succ:
            glb_pred_correct['mean_dist'].append(mean)
            glb_pred_correct['median_dist'].append(median)
            glb_pred_correct['max_dist'].append(max)
            glb_pred_correct['min_dist'].append(min)
        else:
            glb_pred_incorrect['mean_dist'].append(mean)
            glb_pred_incorrect['median_dist'].append(median)
            glb_pred_incorrect['max_dist'].append(max)
            glb_pred_incorrect['min_dist'].append(min)

        # what are the sample's relevant training indices? we need to save them
        original_neighbors_indices = all_neighbor_indices[i, :k]
        adv_dists_to_orig = all_neighbor_dists_adv[i, original_neighbors_indices]

        # fill local
        for ki in range(k):
            adv[ki]['dist'].append(adv_dists_to_orig[ki])
            if attack_succ:
                adv_succ[ki]['dist'].append(adv_dists_to_orig[ki])
            else:
                adv_fail[ki]['dist'].append(adv_dists_to_orig[ki])

        # fill global
        adv_mean   = np.mean(adv_dists_to_orig)
        adv_median = np.median(adv_dists_to_orig)
        adv_max    = np.max(adv_dists_to_orig)
        adv_min    = np.min(adv_dists_to_orig)

        glb_adv['mean_dist'].append(adv_mean)
        glb_adv['median_dist'].append(adv_median)
        glb_adv['max_dist'].append(adv_max)
        glb_adv['min_dist'].append(adv_min)
        glb_adv['mean_dist_ratio'].append(adv_mean / mean)
        if attack_succ:
            glb_adv_succ['mean_dist'].append(adv_mean)
            glb_adv_succ['median_dist'].append(adv_median)
            glb_adv_succ['max_dist'].append(adv_max)
            glb_adv_succ['min_dist'].append(adv_min)
            glb_adv_succ['mean_dist_ratio'].append(adv_mean / mean)
        else:
            glb_adv_fail['mean_dist'].append(adv_mean)
            glb_adv_fail['median_dist'].append(adv_median)
            glb_adv_fail['max_dist'].append(adv_max)
            glb_adv_fail['min_dist'].append(adv_min)
            glb_adv_fail['mean_dist_ratio'].append(adv_mean / mean)

    # summarizing
    all_pred = []
    all_pred_correct = []
    all_pred_incorrect = []
    all_adv = []
    all_adv_succ = []
    all_adv_fail = []
    for ki in range(k):
        all_pred.extend(pred[ki]['dist'])
        all_pred_correct.extend(pred_correct[ki]['dist'])
        all_pred_incorrect.extend(pred_incorrect[ki]['dist'])
        all_adv.extend(adv[ki]['dist'])
        all_adv_succ.extend(adv_succ[ki]['dist'])
        all_adv_fail.extend(adv_fail[ki]['dist'])

    rangee = range_dict[FLAGS.dataset][FLAGS.attack]

    # plotting
    plt.rcParams['interactive'] = False
    plt.figure(1)
    plt.hist(all_pred, range=rangee, label='original', alpha=0.25, bins=300, density=True)
    plt.hist(all_adv, range=rangee, label='adversarial', alpha=0.25, bins=300, density=True)
    plt.title('{} attacked by {}'.format(FLAGS.dataset, FLAGS.attack))
    plt.legend(loc='upper right')
    plt.ylabel('distance from original neighbors')
    plt.savefig('{}_{}_dist_from_orig_neighbors.png'.format(FLAGS.dataset, FLAGS.attack), dpi=350)

    fig, axes = plt.subplots(nrows=3, ncols=2, num=2, figsize=(10, 10))
    fig.suptitle('Distance statistics for {} attacked by {}'.format(FLAGS.dataset, FLAGS.attack), y=1.0)
    axes[0, 0].hist(glb_pred['mean_dist'], range=rangee, label='mean distance', alpha=0.25, bins=300, density=True)
    axes[0, 0].hist(glb_adv['mean_dist'], range=rangee, label='mean distance', alpha=0.25, bins=300, density=True)
    axes[0, 0].legend()
    axes[0, 0].set_ylabel('NN distances')
    axes[0, 1].hist(glb_pred['median_dist'], range=rangee, label='median distance', alpha=0.25, bins=300, density=True)
    axes[0, 1].hist(glb_adv['median_dist'], range=rangee, label='median distance', alpha=0.25, bins=300, density=True)
    axes[0, 1].legend()
    axes[0, 1].set_ylabel('NN distances')
    axes[1, 0].hist(glb_pred['max_dist'], range=rangee, label='max distance', alpha=0.25, bins=300, density=True)
    axes[1, 0].hist(glb_adv['max_dist'], range=rangee, label='max distance', alpha=0.25, bins=300, density=True)
    axes[1, 0].legend()
    axes[1, 0].set_ylabel('NN distances')
    axes[1, 1].hist(glb_pred['min_dist'], range=rangee, label='min distance', alpha=0.25, bins=300, density=True)
    axes[1, 1].hist(glb_adv['min_dist'], range=rangee, label='min distance', alpha=0.25, bins=300, density=True)
    axes[1, 1].legend()
    axes[1, 1].set_ylabel('NN distances')
    axes[2, 0].hist(glb_adv['mean_dist_ratio'], range=rangee, label='adv/orig mean distance ratio', bins=50, density=True)
    axes[2, 0].set_ylabel('NN distances')
    axes[2, 0].legend()
    plt.tight_layout()
    plt.savefig('{}_{}_dist_from_orig_neighbors_stats.png'.format(FLAGS.dataset, FLAGS.attack), dpi=350)

