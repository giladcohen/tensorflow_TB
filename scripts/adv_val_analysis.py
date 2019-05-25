from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import imageio

from tensorflow.python.platform import flags
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
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
flags.DEFINE_bool('targeted', False, 'whether or not the adversarial attack is targeted')
flags.DEFINE_string('characteristics', 'nnif', 'type of defence')
flags.DEFINE_integer('max_indices', 800, 'maximum number of helpful indices to use in NNIF detection')



test_val_set = FLAGS.set == 'val'

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
    ARCH_NAME = 'model_svhn'
    CHECKPOINT_NAME = 'svhn/log_120519_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1257'
    LABEL_SMOOTHING = 0.1
else:
    raise AssertionError('dataset {} not supported'.format(FLAGS.dataset))

superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# get records from training
model_dir  = os.path.join('/data/gilad/logs/influence', CHECKPOINT_NAME)
attack_dir = os.path.join(model_dir, FLAGS.attack)
if FLAGS.targeted:
    attack_dir = attack_dir + '_targeted'
characteristics_dir = os.path.join(attack_dir, FLAGS.characteristics)

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
    print('Crafting {} val noisy samples.'.format(FLAGS.dataset))
    X_val_noisy = get_noisy_samples(X_val, std=STDEVS['val'][FLAGS.dataset][FLAGS.attack])
    np.save(noisy_file, X_val_noisy)

noisy_file = os.path.join(attack_dir, 'X_test_noisy.npy')
if os.path.isfile(noisy_file):
    print('Loading {} noisy samples from {}'.format(FLAGS.dataset, noisy_file))
    X_test_noisy = np.load(noisy_file)
else:
    print('Crafting {} test noisy samples.'.format(FLAGS.dataset))
    X_test_noisy = get_noisy_samples(X_test, std=STDEVS['test'][FLAGS.dataset][FLAGS.attack])
    np.save(noisy_file, X_test_noisy)

# print stats for val
for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_val, X_val_noisy, X_val_adv]):
    # acc = model_eval(sess, x, y, logits, subset, y_val, args=eval_params)
    # print("Model accuracy on the %s val set: %0.2f%%" % (s_type, 100 * acc))
    # Compute and display average perturbation sizes
    if not s_type == 'normal':
        # print for test:
        diff    = subset.reshape((len(subset), -1)) - X_val.reshape((len(subset), -1))
        l2_diff = np.linalg.norm(diff, axis=1).mean()
        print("Average L-2 perturbation size of the %s val set: %0.4f" % (s_type, l2_diff))

# print stats for test
for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_test, X_test_noisy, X_test_adv]):
    # acc = model_eval(sess, x, y, logits, subset, y_test, args=eval_params)
    # print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc))
    # Compute and display average perturbation sizes
    if not s_type == 'normal':
        # print for test:
        diff    = subset.reshape((len(subset), -1)) - X_test.reshape((len(subset), -1))
        l2_diff = np.linalg.norm(diff, axis=1).mean()
        print("Average L-2 perturbation size of the %s test set: %0.4f" % (s_type, l2_diff))

# Refine the normal, noisy and adversarial sets to only include samples for
# which the original version was correctly classified by the model
# val_inds_correct  = np.where(x_val_preds == y_val_sparse)[0]
# print("Number of correctly val predict images: %s" % (len(val_inds_correct)))
# X_val              = X_val[val_inds_correct]
# X_val_noisy        = X_val_noisy[val_inds_correct]
# X_val_adv          = X_val_adv[val_inds_correct]
# x_val_preds        = x_val_preds[val_inds_correct]
# x_val_features     = x_val_features[val_inds_correct]
# x_val_preds_adv    = x_val_preds_adv[val_inds_correct]
# x_val_features_adv = x_val_features_adv[val_inds_correct]
# y_val              = y_val[val_inds_correct]
# y_val_sparse       = y_val_sparse[val_inds_correct]
#
# test_inds_correct = np.where(x_test_preds == y_test_sparse)[0]
# print("Number of correctly test predict images: %s" % (len(test_inds_correct)))
# X_test              = X_test[test_inds_correct]
# X_test_noisy        = X_test_noisy[test_inds_correct]
# X_test_adv          = X_test_adv[test_inds_correct]
# x_test_preds        = x_test_preds[test_inds_correct]
# x_test_features     = x_test_features[test_inds_correct]
# x_test_preds_adv    = x_test_preds_adv[test_inds_correct]
# x_test_features_adv = x_test_features_adv[test_inds_correct]
# y_test              = y_test[test_inds_correct]
# y_test_sparse       = y_test_sparse[test_inds_correct]

print("X_val: "       , X_val.shape)
print("X_val_noisy: " , X_val_noisy.shape)
print("X_val_adv: "   , X_val_adv.shape)

print("X_test: "      , X_test.shape)
print("X_test_noisy: ", X_test_noisy.shape)
print("X_test_adv: "  , X_test_adv.shape)

sub_relevant_indices = [ind for ind in info[FLAGS.set] if info[FLAGS.set][ind]['attack_succ']]
relevant_indices     = [info[FLAGS.set][ind]['global_index'] for ind in sub_relevant_indices]

real = {}
pred = {}
pred_correct = {}
pred_incorrect = {}
adv  = {}
for i in range(feeder.get_val_size()):
    real[i]           = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
    pred[i]           = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
    pred_correct[i]   = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
    pred_incorrect[i] = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}
    adv[i]            = {'+dist': [], '+rank': [], '-dist': [], '-rank': []}

    # is_correct = y_val_sparse[i] == x_val_preds[i]  # did the model predicted right?
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

    # collect adv - only for real = pred != adv
    if net_succ and attack_succ:
        helpful_ranks = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'helpful_ranks.npy'))
        helpful_dists = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'helpful_dists.npy'))
        harmful_ranks = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'harmful_ranks.npy'))
        harmful_dists = np.load(os.path.join(index_dir, 'adv', FLAGS.attack, 'harmful_dists.npy'))
        for k in range(helpful_ranks.shape[0]):
            adv[k]['+rank'].append(helpful_ranks[k])
            adv[k]['+dist'].append(helpful_dists[k])
            adv[k]['-rank'].append(harmful_ranks[k])
            adv[k]['-dist'].append(harmful_dists[k])


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
    for i in range(FLAGS.max_indices):
        all_real.extend(real[i]['+rank'])
        all_pred.extend(pred[i]['+rank'])
        all_pred_correct.extend(pred_correct[i]['+rank'])
        all_pred_incorrect.extend(pred_incorrect[i]['+rank'])
        all_adv.extend(adv[i]['+rank'])

    all_values = all_real + all_pred + all_adv
    rangee = (np.min(all_values), np.max(all_values))

    ax1.hist(all_real, range=rangee, label='real', alpha=0.25, bins=300, density=True)
    ax1.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=300, density=True)
    ax1.set_title('Deepfool')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('+rank PDF')
    # ax1.set_xlabel('+rank', horizontalalignment='right')
    ax1.set_xlim(0, 10000)

    # + dist
    ax2 = fig.add_subplot(312)
    all_real = []
    all_pred = []
    all_pred_correct = []
    all_pred_incorrect = []
    all_adv = []
    for i in range(FLAGS.max_indices):
        all_real.extend(real[i]['+dist'])
        all_pred.extend(pred[i]['+dist'])
        all_pred_correct.extend(pred_correct[i]['+dist'])
        all_pred_incorrect.extend(pred_incorrect[i]['+dist'])
        all_adv.extend(adv[i]['+dist'])

    all_values = all_real + all_pred + all_adv
    rangee = (np.min(all_values), np.max(all_values))

    ax2.hist(all_real, range=rangee, label='real', alpha=0.25, bins=200, density=True)
    ax2.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=200, density=True)
    ax2.hist(all_pred_incorrect, range=rangee, label='pred(incorrect)', alpha=0.25, bins=200, density=True)
    ax2.legend(loc='upper center')
    ax2.set_ylabel('+dist PDF')
    # ax2.set_xlabel('+dist', horizontalalignment='right')
    ax2.set_xlim(0, 8)

    # - dist
    ax3 = fig.add_subplot(313)
    all_real = []
    all_pred = []
    all_pred_correct = []
    all_pred_incorrect = []
    all_adv = []
    for i in range(FLAGS.max_indices):
        all_real.extend(real[i]['-dist'])
        all_pred.extend(pred[i]['-dist'])
        all_pred_correct.extend(pred_correct[i]['-dist'])
        all_pred_incorrect.extend(pred_incorrect[i]['-dist'])
        all_adv.extend(adv[i]['-dist'])

    all_values = all_real + all_pred + all_adv
    rangee = (np.min(all_values), np.max(all_values))

    ax3.hist(all_real, range=rangee, label='real', alpha=0.25, bins=200, density=True)
    ax3.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=200, density=True)
    ax3.legend(loc='upper right')
    ax3.set_ylabel('-dist PDF')
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
    for i in range(FLAGS.max_indices):
        all_real.extend(real[i]['+rank'])
        all_pred.extend(pred[i]['+rank'])
        all_pred_correct.extend(pred_correct[i]['+rank'])
        all_pred_incorrect.extend(pred_incorrect[i]['+rank'])
        all_adv.extend(adv[i]['+rank'])

    all_values = all_real + all_pred + all_adv
    rangee = (np.min(all_values), np.max(all_values))

    ax1.hist(all_real, range=rangee, label='real', alpha=0.25, bins=300, density=True)
    ax1.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=300, density=True)
    ax1.set_title('Carlini-Wagner')
    ax1.legend(loc='upper right')
    ax1.set_ylabel('+rank PDF')
    # ax1.set_xlabel('+rank', horizontalalignment='right')
    ax1.set_xlim(0, 10000)

    # + dist
    ax2 = fig.add_subplot(312)
    all_real = []
    all_pred = []
    all_pred_correct = []
    all_pred_incorrect = []
    all_adv = []
    for i in range(FLAGS.max_indices):
        all_real.extend(real[i]['+dist'])
        all_pred.extend(pred[i]['+dist'])
        all_pred_correct.extend(pred_correct[i]['+dist'])
        all_pred_incorrect.extend(pred_incorrect[i]['+dist'])
        all_adv.extend(adv[i]['+dist'])

    all_values = all_real + all_pred + all_adv
    rangee = (np.min(all_values), np.max(all_values))

    ax2.hist(all_real, range=rangee, label='real', alpha=0.25, bins=200, density=True)
    ax2.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=200, density=True)
    ax2.hist(all_pred_incorrect, range=rangee, label='pred(incorrect)', alpha=0.25, bins=200, density=True)
    ax2.legend(loc='upper right')
    ax2.set_ylabel('+dist PDF')
    # ax2.set_xlabel('+dist', horizontalalignment='right')
    ax2.set_xlim(0, 5)

    # - dist
    ax3 = fig.add_subplot(313)
    all_real = []
    all_pred = []
    all_pred_correct = []
    all_pred_incorrect = []
    all_adv = []
    for i in range(FLAGS.max_indices):
        all_real.extend(real[i]['-dist'])
        all_pred.extend(pred[i]['-dist'])
        all_pred_correct.extend(pred_correct[i]['-dist'])
        all_pred_incorrect.extend(pred_incorrect[i]['-dist'])
        all_adv.extend(adv[i]['-dist'])

    all_values = all_real + all_pred + all_adv
    rangee = (np.min(all_values), np.max(all_values))

    ax3.hist(all_real, range=rangee, label='real', alpha=0.25, bins=200, density=True)
    ax3.hist(all_adv, range=rangee, label='adv', alpha=0.25, bins=200, density=True)
    ax3.legend(loc='upper right')
    ax3.set_ylabel('-dist PDF')
    # ax2.set_xlabel('+dist', horizontalalignment='right')
    ax3.set_xlim(3, 8)

    plt.savefig('carlini_wagner_rank_and_dist_hists.png', dpi=350)


# def plot_hists(max_index, f):
#     """
#     :param max_index: max index to collect
#     :param f: field: +dist/-dist/+rank/-rank
#     :return: None
#     """
#     # start creating statistics
#     # just for the most helpful/harmful
#     all_real           = []
#     all_pred           = []
#     all_pred_correct   = []
#     all_pred_incorrect = []
#     all_adv            = []
#
#     for i in range(max_index):
#         all_real.extend(real[i][f])
#         all_pred.extend(pred[i][f])
#         all_pred_correct.extend(pred_correct[i][f])
#         all_pred_incorrect.extend(pred_incorrect[i][f])
#         all_adv.extend(adv[i][f])
#
#     all_values = all_real + all_pred + all_adv
#     rangee = (np.min(all_values), np.max(all_values))
#
#     plt.hist(all_real          , range=rangee, label='real'           , alpha=0.25, bins=50, density=True)
#     plt.hist(all_pred_correct  , range=rangee, label='pred(correct)'  , alpha=0.25, bins=50, density=True)
#     plt.hist(all_pred_incorrect, range=rangee, label='pred(incorrect)', alpha=0.25, bins=50, density=True)
#     plt.hist(all_adv           , range=rangee, label='adv'            , alpha=0.25, bins=50, density=True)
#     title = 'top {} samples for {}'.format(max_index, f)
#     plt.title(title)
#     plt.legend(loc='upper right')
#     plt.savefig(title, dpi=350)
#     plt.close()
#
#     plt.hist(all_real, range=rangee, label='real', alpha=0.5, bins=50, density=True)
#     plt.hist(all_adv , range=rangee, label='adv' , alpha=0.5, bins=50, density=True)
#     title = 'top {} samples for {} - just real vs adv'.format(max_index, f)
#     plt.title(title)
#     plt.legend(loc='upper right')
#     plt.savefig(title, dpi=350)
#     plt.close()
#
# for max_index in [800]:
#     for f in ['+rank', '-rank', '+dist', '-dist']:
#         plot_hists(max_index=max_index, f=f)

