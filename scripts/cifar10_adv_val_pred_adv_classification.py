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
from sklearn import metrics

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_name', 'cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000', 'checkpoint name')
flags.DEFINE_string('set', 'val', 'val or test set to evaluate')
flags.DEFINE_bool('use_train_mini', False, 'Whether or not to use 5000 training samples instead of 49000')
flags.DEFINE_string('dataset', 'cifar10', 'datasset: cifar10/100')

test_val_set = FLAGS.set == 'val'

# cifar10 classes
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
else:
    raise AssertionError('dataset {} not supported'.format(FLAGS.dataset))

# Get CIFAR-10 data
cifar10_input.maybe_download_and_extract()

superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# get records from training
model_dir     = os.path.join('/data/gilad/logs/influence', FLAGS.checkpoint_name)

mini_train_inds = None  # np.load(os.path.join(model_dir, 'train_mini_indices.npy'))
val_indices     = np.load(os.path.join(model_dir, 'val_indices.npy'))

feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
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

sub_relevant_indices = [ind for ind in info[FLAGS.set] if info[FLAGS.set][ind]['net_succ']]  #TODO(remove after bug fix)
relevant_indices     = [info[FLAGS.set][ind]['global_index'] for ind in sub_relevant_indices]

# guessing a threshold rank/dist for the "adv"/"non-adversarial" decision
# max_indices        = np.arange(600, 1001, 10)
max_indices        = np.array([800])
rank_values        = np.arange(10, 10000, 10)
dist_values        = np.arange(3.7, 3.7, 0.01)
rank_x_dist_values = np.arange(18000, 18000, 100)

def rt():
    return np.zeros(shape=(max_indices.shape[0], rank_values.shape[0]))
def dt():
    return np.zeros(shape=(max_indices.shape[0], dist_values.shape[0]))
def rxd():
    return np.zeros(shape=(max_indices.shape[0], rank_x_dist_values.shape[0]))


# metrics = {
#     'rank_thd'   : {'tp': rt() , 'fp': rt() , 'tn': rt() , 'fn': rt() , 'precision': rt() , 'recall': rt() , 'accuracy': rt()},
#     'dist_thd'   : {'tp': dt() , 'fp': dt() , 'tn': dt() , 'fn': dt() , 'precision': dt() , 'recall': dt() , 'accuracy': dt()},
#     'rank_x_dist': {'tp': rxd(), 'fp': rxd(), 'tn': rxd(), 'fn': rxd(), 'precision': rxd(), 'recall': rxd(), 'accuracy': rxd()}
# }
#
# This loop ends up collecting metrics for the entire set, each for different max_indices and rank_value
# for i, sub_index in enumerate(sub_relevant_indices):
#     global_index = feeder.val_inds[sub_index]
#     print('start working on sample i={}. global_index={}'.format(i, global_index))
#     assert global_index == relevant_indices[i]
#
#     real_label = y_val_sparse[sub_index]
#     pred_label = x_val_preds[sub_index]
#     adv_label  = x_val_preds_adv[sub_index]
#
#     net_succ    = info[FLAGS.set][sub_index]['net_succ']
#     attack_succ = info[FLAGS.set][sub_index]['attack_succ']
#
#     if attack_succ:
#         assert pred_label != adv_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)
#     if net_succ:
#         assert pred_label == real_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)
#
#     index_dir = os.path.join(model_dir, 'val', 'val_index_{}'.format(global_index))
#
#     # collect pred (negative)
#     scores_dir = 'real' if net_succ else 'pred'
#     pred_helpful_ranks = np.load(os.path.join(index_dir, scores_dir, 'helpful_ranks.npy'))
#     pred_helpful_dists = np.load(os.path.join(index_dir, scores_dir, 'helpful_dists.npy'))
#     pred_harmful_ranks = np.load(os.path.join(index_dir, scores_dir, 'harmful_ranks.npy'))
#     pred_harmful_dists = np.load(os.path.join(index_dir, scores_dir, 'harmful_dists.npy'))
#
#     if attack_succ:
#         scores_dir = 'adv'
#         adv_helpful_ranks = np.load(os.path.join(index_dir, scores_dir, 'helpful_ranks.npy'))
#         adv_helpful_dists = np.load(os.path.join(index_dir, scores_dir, 'helpful_dists.npy'))
#         adv_harmful_ranks = np.load(os.path.join(index_dir, scores_dir, 'harmful_ranks.npy'))
#         adv_harmful_dists = np.load(os.path.join(index_dir, scores_dir, 'harmful_dists.npy'))
#
#     for m, max_index in enumerate(max_indices):
#         # rank_thd
#         for r, rank_thd in enumerate(rank_values):
#             # calculate rank_thd for negative pred
#             ranks_mean = pred_helpful_ranks[0:max_index].mean()
#             if ranks_mean < rank_thd:
#                 metrics['rank_thd']['tn'][m, r] += 1
#             else:
#                 metrics['rank_thd']['fn'][m, r] += 1
#
#             if attack_succ:
#                 # calculate rank_thd for positive adv
#                 ranks_mean = adv_helpful_ranks[0:max_index].mean()
#                 if ranks_mean >= rank_thd:
#                     metrics['rank_thd']['tp'][m, r] += 1
#                 else:
#                     metrics['rank_thd']['fp'][m, r] += 1
#
#         # dist_thd
#         for d, dist_thd in enumerate(dist_values):
#             # calculate dist_thd for negative pred
#             dists_mean = pred_helpful_dists[0:max_index].mean()
#             if dists_mean < dist_thd:
#                 metrics['dist_thd']['tn'][m, d] += 1
#             else:
#                 metrics['dist_thd']['fn'][m, d] += 1
#
#             if attack_succ:
#                 # calculate dist_thd for positive adv
#                 dists_mean = adv_helpful_dists[0:max_index].mean()
#                 if dists_mean >= dist_thd:
#                     metrics['dist_thd']['tp'][m, d] += 1
#                 else:
#                     metrics['dist_thd']['fp'][m, d] += 1
#
#         # rank_x_dist
#         for rd, rank_x_dist_thd in enumerate(rank_x_dist_values):
#             # calculate rank_x_dist_thd for negative pred
#             rxd_mean = (pred_helpful_ranks[0:max_index] * pred_helpful_dists[0:max_index]).mean()
#             if rxd_mean < rank_x_dist_thd:
#                 metrics['rank_x_dist']['tn'][m, rd] += 1
#             else:
#                 metrics['rank_x_dist']['fn'][m, rd] += 1
#
#             if attack_succ:
#                 # calculate rank_x_dist_thd for positive adv
#                 rxd_mean = (adv_helpful_ranks[0:max_index] * adv_helpful_dists[0:max_index]).mean()
#                 if rxd_mean >= rank_x_dist_thd:
#                     metrics['rank_x_dist']['tp'][m, rd] += 1
#                 else:
#                     metrics['rank_x_dist']['fp'][m, rd] += 1
#
#     del pred_helpful_ranks, pred_helpful_dists, pred_harmful_ranks, pred_harmful_dists
#     if attack_succ:
#         del adv_helpful_ranks, adv_helpful_dists, adv_harmful_ranks, adv_harmful_dists

y_true  = []
y_score = []
for i, sub_index in enumerate(sub_relevant_indices):
    global_index = feeder.val_inds[sub_index]
    print('start working on sample i={}. global_index={}'.format(i, global_index))
    assert global_index == relevant_indices[i]

    real_label = y_val_sparse[sub_index]
    pred_label = x_val_preds[sub_index]
    adv_label  = x_val_preds_adv[sub_index]

    net_succ    = info[FLAGS.set][sub_index]['net_succ']
    assert net_succ  #TODO(remove after bug fix)
    attack_succ = info[FLAGS.set][sub_index]['attack_succ']

    if attack_succ:
        assert pred_label != adv_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)
    if net_succ:
        assert pred_label == real_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)

    index_dir = os.path.join(model_dir, 'val', 'val_index_{}'.format(global_index))

    # collect pred (negative)
    scores_dir = 'real' if net_succ else 'pred'
    pred_helpful_ranks = np.load(os.path.join(index_dir, scores_dir, 'helpful_ranks.npy'))
    pred_helpful_dists = np.load(os.path.join(index_dir, scores_dir, 'helpful_dists.npy'))
    pred_harmful_ranks = np.load(os.path.join(index_dir, scores_dir, 'harmful_ranks.npy'))
    pred_harmful_dists = np.load(os.path.join(index_dir, scores_dir, 'harmful_dists.npy'))

    if attack_succ:
        scores_dir = 'adv'
        adv_helpful_ranks = np.load(os.path.join(index_dir, scores_dir, 'helpful_ranks.npy'))
        adv_helpful_dists = np.load(os.path.join(index_dir, scores_dir, 'helpful_dists.npy'))
        adv_harmful_ranks = np.load(os.path.join(index_dir, scores_dir, 'harmful_ranks.npy'))
        adv_harmful_dists = np.load(os.path.join(index_dir, scores_dir, 'harmful_dists.npy'))

    # start with the real label
    y_true.append(0)
    ranks_mean = pred_helpful_ranks[0:800].mean()
    y_score.append(ranks_mean / 50000)

    # add an adv label if attack succeeded
    if attack_succ:
        y_true.append(1)
        ranks_mean = adv_helpful_ranks[0:800].mean()
        y_score.append(ranks_mean / 50000)

y_true  = np.asarray(y_true)
y_score = np.asarray(y_score)
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
auc = metrics.roc_auc_score(y_true, y_score)















for key in metrics.keys():
    if key == 'rank_thd':
        values = rank_values
    elif key == 'dist_thd':
        values = dist_values
    elif key == 'rank_x_dist':
        values = rank_x_dist_values
    else:
        raise AssertionError
    for m in range(len(max_indices)):
        for r in range(len(values)):
            tp, fp, tn, fn = metrics[key]['tp'][m, r], metrics[key]['fp'][m, r], \
                             metrics[key]['tn'][m, r], metrics[key]['fn'][m, r]
            metrics[key]['precision'][m, r] = tp/(tp+fp)
            metrics[key]['recall'][m, r]    = tp/(tp+fn)  # = TPR
            metrics[key]['fpr'][m, r]       = fp/(tn+fp)  # = FPR
            metrics[key]['accuracy'][m, r]  = (tp+tn)/(tp+tn+fp+fn)

# plotting accuracy metric for all methods
# rank_thd
X, Y = np.meshgrid(max_indices, rank_values)
Z = np.zeros((len(rank_values), len(max_indices)))
for i in range(len(rank_values)):
    for j in range(len(max_indices)):
        Z[i, j] = metrics['rank_thd']['accuracy'][j, i]
c = plt.pcolor(X, Y, Z, cmap='RdBu', edgecolors='face')
plt.xlabel('num of top helpful training samples')
plt.ylabel('rank threshold')
plt.title('adversarial image classification accuracy')
plt.colorbar(c)
plt.show()

# dist_thd
# X, Y = np.meshgrid(max_indices, dist_values)
# Z = np.zeros((len(dist_values), len(max_indices)))
# for i in range(len(dist_values)):
#     for j in range(len(max_indices)):
#         Z[i, j] = metrics['dist_thd']['accuracy'][j, i]
# c = plt.pcolor(X, Y, Z, cmap='RdBu', edgecolors='face')
# plt.xlabel('num of top helpful training samples')
# plt.ylabel('distance threshold')
# plt.title('adversarial image classification accuracy')
# plt.colorbar(c)
# plt.show()

# rank_dist_thd
# X, Y = np.meshgrid(max_indices, rank_x_dist_values)
# Z = np.zeros((len(rank_x_dist_values), len(max_indices)))
# for i in range(len(rank_x_dist_values)):
#     for j in range(len(max_indices)):
#         Z[i, j] = metrics['rank_x_dist']['accuracy'][j, i]
# c = plt.pcolor(X, Y, Z, cmap='RdBu', edgecolors='face')
# plt.xlabel('num of top helpful training samples')
# plt.ylabel('rank x dist threshold')
# plt.title('adversarial image classification accuracy')
# plt.colorbar(c)
# plt.show()
