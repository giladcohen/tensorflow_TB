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

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_name', 'log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000', 'checkpoint name')
flags.DEFINE_string('set', 'val', 'val or test set to evaluate')
flags.DEFINE_bool('use_train_mini', False, 'Whether or not to use 5000 training samples instead of 49000')

test_val_set = FLAGS.set == 'val'

# cifar-10 classes
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

# Get CIFAR-10 data
cifar10_input.maybe_download_and_extract()

superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# get records from training
model_dir     = os.path.join('/data/gilad/logs/influence', FLAGS.checkpoint_name)

mini_train_inds = None  # np.load(os.path.join(model_dir, 'train_mini_indices.npy'))
val_indices     = np.load(os.path.join(model_dir, 'val_indices.npy'))

feeder = MyFeederValTest(rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                         test_val_set=False, mini_train_inds=mini_train_inds)

# get the data
X_train, y_train       = feeder.train_indices(range(feeder.get_train_size()))
X_val, y_val           = feeder.val_indices(range(feeder.get_val_size()))
X_test, y_test         = feeder.test_data, feeder.test_label  # getting the real test set
y_train_sparse         = y_train.argmax(axis=-1).astype(np.int32)
y_val_sparse           = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse          = y_test.argmax(axis=-1).astype(np.int32)

# predict labels from trainset
if FLAGS.use_train_mini:
    train_preds_file    = os.path.join(model_dir, 'x_train_mini_preds.npy')
    train_features_file = os.path.join(model_dir, 'x_train_mini_features.npy')
else:
    train_preds_file    = os.path.join(model_dir, 'x_train_preds.npy')
    train_features_file = os.path.join(model_dir, 'x_train_features.npy')
x_train_preds = np.load(train_preds_file)
x_train_features = np.load(train_features_file)

# predict labels from validation set
x_val_preds    = np.load(os.path.join(model_dir, 'x_val_preds.npy'))
x_val_features = np.load(os.path.join(model_dir, 'x_val_features.npy'))

# predict labels from test set
x_test_preds    = np.load(os.path.join(model_dir, 'x_test_preds.npy'))
x_test_features = np.load(os.path.join(model_dir, 'x_test_features.npy'))

# predict labels from adv validation set
X_val_adv          = np.load(os.path.join(model_dir, 'X_val_adv.npy'))
x_val_preds_adv    = np.load(os.path.join(model_dir, 'x_val_preds_adv.npy'))
x_val_features_adv = np.load(os.path.join(model_dir, 'x_val_features_adv.npy'))

# predict labels from adv test set
X_test_adv = np.load(os.path.join(model_dir, 'X_test_adv.npy'))
x_test_preds_adv = np.load(os.path.join(model_dir, 'x_test_preds_adv.npy'))
x_test_features_adv = np.load(os.path.join(model_dir, 'x_test_features_adv.npy'))

# quick computations
train_acc    = np.mean(y_train_sparse == x_train_preds)
val_acc      = np.mean(y_val_sparse   == x_val_preds)
test_acc     = np.mean(y_test_sparse  == x_test_preds)
val_adv_acc  = np.mean(y_val_sparse   == x_val_preds_adv)
test_adv_acc = np.mean(y_test_sparse  == x_test_preds_adv)
print('train set acc: {}\nvalidation set acc: {}\ntest set acc: {}'.format(train_acc, val_acc, test_acc))
print('adversarial validation set acc: {}\nadversarial test set acc: {}'.format(val_adv_acc, test_adv_acc))

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

info_file = os.path.join(model_dir, 'info.pkl')
if not os.path.isfile(info_file):
    print('saving info as pickle to {}'.format(info_file))
    with open(info_file, 'wb') as handle:
        pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print('loading info as pickle from {}'.format(info_file))
    with open(info_file, 'rb') as handle:
        info_old = pickle.load(handle)
    assert info == info_old

sub_relevant_indices = [ind for ind in info[FLAGS.set]]
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
        helpful_ranks = np.load(os.path.join(index_dir, 'adv', 'helpful_ranks.npy'))
        helpful_dists = np.load(os.path.join(index_dir, 'adv', 'helpful_dists.npy'))
        harmful_ranks = np.load(os.path.join(index_dir, 'adv', 'harmful_ranks.npy'))
        harmful_dists = np.load(os.path.join(index_dir, 'adv', 'harmful_dists.npy'))
        for k in range(helpful_ranks.shape[0]):
            adv[k]['+rank'].append(helpful_ranks[k])
            adv[k]['+dist'].append(helpful_dists[k])
            adv[k]['-rank'].append(harmful_ranks[k])
            adv[k]['-dist'].append(harmful_dists[k])

def plot_hists(max_index, f):
    """
    :param max_index: max index to collect
    :param f: field: +dist/-dist/+rank/-rank
    :return: None
    """
    # start creating statistics
    # just for the most helpful/harmful
    all_real           = []
    all_pred           = []
    all_pred_correct   = []
    all_pred_incorrect = []
    all_adv            = []

    for i in range(max_index):
        all_real.extend(real[i][f])
        all_pred.extend(pred[i][f])
        all_pred_correct.extend(pred_correct[i][f])
        all_pred_incorrect.extend(pred_incorrect[i][f])
        all_adv.extend(adv[i][f])

    all_values = all_real + all_pred + all_adv
    rangee = (np.min(all_values), np.max(all_values))

    plt.hist(all_real          , range=rangee, label='real'           , alpha=0.25, bins=50, density=True)
    plt.hist(all_pred_correct  , range=rangee, label='pred(correct)'  , alpha=0.25, bins=50, density=True)
    plt.hist(all_pred_incorrect, range=rangee, label='pred(incorrect)', alpha=0.25, bins=50, density=True)
    plt.hist(all_adv           , range=rangee, label='adv'            , alpha=0.25, bins=50, density=True)
    title = 'top {} samples for {}'.format(max_index, f)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(title, dpi=350)
    plt.close()

    plt.hist(all_real, range=rangee, label='real', alpha=0.5, bins=50, density=True)
    plt.hist(all_adv , range=rangee, label='adv' , alpha=0.5, bins=50, density=True)
    title = 'top {} samples for {} - just real vs adv'.format(max_index, f)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(title, dpi=350)
    plt.close()


for max_index in [1, 250, 500, 750, 1000]:
    for f in ['+rank', '-rank', '+dist', '-dist']:
        plot_hists(max_index=max_index, f=f)

