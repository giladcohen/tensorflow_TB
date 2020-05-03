from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import os
import imageio

import darkon.darkon as darkon
from sklearn.neighbors import NearestNeighbors

from tensorflow.python.platform import flags
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow_TB.lib.datasets.influence_feeder_val_test import MyFeederValTest

# tf.enable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cifar10', 'dataset: cifar10/100 or svhn')
flags.DEFINE_string('attack', 'deepfool', 'adversarial attack: deepfool, jsma, cw')
flags.DEFINE_bool('targeted', False, 'whether or not the adversarial attack is targeted')
flags.DEFINE_integer('k_nearest', 25, 'number of nearest neighbors for plots')

flags.DEFINE_string('mode', 'null', 'to bypass pycharm bug')
flags.DEFINE_string('port', 'null', 'to bypass pycharm bug')


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
    CHECKPOINT_NAME = 'cifar10/log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000'
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
    CHECKPOINT_NAME = 'cifar100/log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01'
elif FLAGS.dataset == 'svhn':
    _classes = (
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    )
    CHECKPOINT_NAME = 'svhn/log_120519_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1257'
else:
    raise AssertionError('dataset {} not supported'.format(FLAGS.dataset))

# Set TF random seed to improve reproducibility
superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# get records from training
model_dir          = os.path.join('/data/gilad/logs/influence', CHECKPOINT_NAME)
attack_dir         = os.path.join(model_dir, FLAGS.attack)
if FLAGS.targeted:
    attack_dir = attack_dir + '_targeted'
plot_dir = os.path.join(attack_dir, 'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

print('loading train mini indices from {}'.format(os.path.join(model_dir, 'train_mini_indices.npy')))
mini_train_inds = None
val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))
feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                         test_val_set=True, mini_train_inds=mini_train_inds)

# get the dataset
X_train     , y_train      = feeder.train_data     , feeder.train_label        # real train set (49k):
X_val       , y_val        = feeder.val_data       , feeder.val_label          # val set (1k)
X_test      , y_test       = feeder.test_data      , feeder.test_label         # test set
y_train_sparse             = y_train.argmax(axis=-1).astype(np.int32)
y_val_sparse               = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse              = y_test.argmax(axis=-1).astype(np.int32)

# if the attack is targeted, fetch the targets
if FLAGS.targeted:
    y_val_targets  = np.load(os.path.join(attack_dir, 'y_val_targets.npy'))
    y_test_targets = np.load(os.path.join(attack_dir, 'y_test_targets.npy'))

# fetch the predictions and embedding vectors
x_train_preds         = np.load(os.path.join(model_dir, 'x_train_preds.npy'))
x_train_features      = np.load(os.path.join(model_dir, 'x_train_features.npy'))

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

# Refine the normal, noisy and adversarial sets to only include samples for
# which the original version was correctly classified by the model
val_inds_correct  = np.where(x_val_preds == y_val_sparse)[0]
print("Number of correctly val predict images: %s" % (len(val_inds_correct)))
# creating map from a regular val index to subset (predicted correctly) val index
val_idx_map = {}
for i in range(len(val_inds_correct)):
    val_idx_map[i] = val_inds_correct[i]

X_val              = X_val[val_inds_correct]
X_val_adv          = X_val_adv[val_inds_correct]
x_val_preds        = x_val_preds[val_inds_correct]
x_val_features     = x_val_features[val_inds_correct]
x_val_preds_adv    = x_val_preds_adv[val_inds_correct]
x_val_features_adv = x_val_features_adv[val_inds_correct]
y_val              = y_val[val_inds_correct]
y_val_sparse       = y_val_sparse[val_inds_correct]

# Fitting the KNN model
print('fitting the KNN model...')
knn = NearestNeighbors(n_neighbors=feeder.get_train_size(), p=2, n_jobs=20)
knn.fit(x_train_features, y_train_sparse)
print('Done fitting the knn model.')

# Fitting the PCA
print('fitting the PCA model...')
if not os.path.exists(os.path.join(plot_dir, 'pca', 'NN_{}'.format(FLAGS.k_nearest))):
    os.makedirs(os.path.join(plot_dir, 'pca', 'NN_{}'.format(FLAGS.k_nearest)))
pca = PCA(n_components=2, svd_solver='full', random_state=rand_gen)
pca.fit(x_train_features)
pca_x_train_embedded   = pca.transform(x_train_features)
pca_x_val_embedded     = pca.transform(x_val_features)
pca_x_val_adv_embedded = pca.transform(x_val_features_adv)
print('Done fitting the PCA model.')

# Fitting the TSNE
# just for making thing faster, saving the tsne embedding in the plot folder
if not os.path.exists(os.path.join(plot_dir, 'tsne', 'NN_{}'.format(FLAGS.k_nearest))):
    os.makedirs(os.path.join(plot_dir, 'tsne', 'NN_{}'.format(FLAGS.k_nearest)))
if not os.path.exists(os.path.join(plot_dir, 'tsne', 'tsne_x_train_embedded.npy')):
    print('fitting the TSNE model...')
    tsne = TSNE(n_components=2)
    x_train_val_features = np.concatenate((x_train_features,
                                           x_val_features,
                                           x_val_features_adv))
    x_train_val_embedded = tsne.fit_transform(x_train_val_features)
    val_offset     = x_train_features.shape[0]
    val_adv_offest = x_train_features.shape[0] + x_val_features.shape[0]

    tsne_x_train_embedded   = x_train_val_embedded[:val_offset]
    tsne_x_val_embedded     = x_train_val_embedded[val_offset:val_adv_offest]
    tsne_x_val_adv_embedded = x_train_val_embedded[val_adv_offest:]
    print('Done fitting the TSNE model.')

    np.save(os.path.join(plot_dir, 'tsne', 'tsne_x_train_embedded.npy')  , tsne_x_train_embedded)
    np.save(os.path.join(plot_dir, 'tsne', 'tsne_x_val_embedded.npy')    , tsne_x_val_embedded)
    np.save(os.path.join(plot_dir, 'tsne', 'tsne_x_val_adv_embedded.npy'), tsne_x_val_adv_embedded)
else:
    tsne_x_train_embedded   = np.load(os.path.join(plot_dir, 'tsne', 'tsne_x_train_embedded.npy'))
    tsne_x_val_embedded     = np.load(os.path.join(plot_dir, 'tsne', 'tsne_x_val_embedded.npy'))
    tsne_x_val_adv_embedded = np.load(os.path.join(plot_dir, 'tsne', 'tsne_x_val_adv_embedded.npy'))

# for key, val in val_idx_map.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
#     global_index = feeder.get_global_index('val', val)
#     if global_index == 31732:
#         print(key)

# for vis_idx in range(len(X_val)):
    vis_idx = 596
    plt.close('all')
    vis_img     = X_val[vis_idx]
    vis_img_adv = X_val_adv[vis_idx]

    vis_features     = np.expand_dims(x_val_features[vis_idx]    , axis=0)
    vis_features_adv = np.expand_dims(x_val_features_adv[vis_idx], axis=0)

    # get the neighbors:
    neighbor_dists    , neighbor_indices     = knn.kneighbors(vis_features)
    neighbor_dists_adv, neighbor_indices_adv = knn.kneighbors(vis_features_adv)

    # get the 50 nearest neighbors pca/tsne embedding
    top_k_neighbors     = neighbor_indices.squeeze(0)[:FLAGS.k_nearest]
    top_k_neighbors_adv = neighbor_indices_adv.squeeze(0)[:FLAGS.k_nearest]

    neighbors_pca_embeddings      = pca_x_train_embedded[top_k_neighbors]
    neighbors_pca_embeddings_adv  = pca_x_train_embedded[top_k_neighbors_adv]
    neighbors_tsne_embeddings     = tsne_x_train_embedded[top_k_neighbors]
    neighbors_tsne_embeddings_adv = tsne_x_train_embedded[top_k_neighbors_adv]

    # get the 50 most helpful training samples
    global_val_index = feeder.get_global_index('val', val_idx_map[vis_idx])
    index_dir = os.path.join(model_dir, 'val', 'val_index_{}'.format(global_val_index))
    scores     = np.load(os.path.join(index_dir, 'real', 'scores.npy'))
    scores_adv = np.load(os.path.join(index_dir, 'adv' , FLAGS.attack, 'scores.npy'))
    top_helpful_indices     = np.argsort(scores)[-FLAGS.k_nearest:]
    top_helpful_indices_adv = np.argsort(scores_adv)[-FLAGS.k_nearest:]

    top_helpful_pca_embeddings      = pca_x_train_embedded[top_helpful_indices]
    top_helpful_pca_embeddings_adv  = pca_x_train_embedded[top_helpful_indices_adv]
    top_helpful_tsne_embeddings     = tsne_x_train_embedded[top_helpful_indices]
    top_helpful_tsne_embeddings_adv = tsne_x_train_embedded[top_helpful_indices_adv]

    # plotting the PCA:
    vis_embeddings     = np.expand_dims(pca_x_val_embedded[vis_idx], axis=0)
    vis_embeddings_adv = np.expand_dims(pca_x_val_adv_embedded[vis_idx], axis=0)

    i=30
    plt.figure(i, (5, 5))
    plt.scatter(vis_embeddings[:, 0]    , vis_embeddings[:, 1]    , s=200, marker='*', c='black', label='normal')
    plt.scatter(vis_embeddings_adv[:, 0], vis_embeddings_adv[:, 1], s=200, marker='X', c='brown',  label='adv')
    plt.legend(loc='upper left')
    plt.ylim([-0.25213, 3.56749])
    plt.xlim([-0.94303, 1.99217])
    plt.tight_layout()
    plt.savefig('teaser_start.png', dpi=300)

    plt.figure(i+1, (5, 5))
    plt.scatter(vis_embeddings[:, 0]    , vis_embeddings[:, 1]    , s=200, marker='*', c='black', label='normal')
    plt.scatter(vis_embeddings_adv[:, 0], vis_embeddings_adv[:, 1], s=200, marker='X', c='brown',  label='adv')
    plt.scatter(neighbors_pca_embeddings[:, 0], neighbors_pca_embeddings[:, 1], s=10, marker='o', c='blue', alpha=0.5, label='normal $k$-NN')
    plt.legend(loc='upper left')
    plt.ylim([-0.25213, 3.56749])
    plt.xlim([-0.94303, 1.99217])
    plt.tight_layout()
    plt.savefig('teaser_start2.png', dpi=300)

    plt.figure(i+2, (5, 5))
    plt.scatter(neighbors_pca_embeddings[:, 0], neighbors_pca_embeddings[:, 1], s=10, marker='o', c='blue', alpha=0.5, label='normal $k$-NN')
    plt.scatter(neighbors_pca_embeddings_adv[:, 0], neighbors_pca_embeddings_adv[:, 1], s=10, marker='v', c='red' , alpha=0.5, label='adv $k$-NN')
    plt.scatter(vis_embeddings[:, 0]    , vis_embeddings[:, 1]    , s=200, marker='*', c='black', label='normal')
    plt.scatter(vis_embeddings_adv[:, 0], vis_embeddings_adv[:, 1], s=200, marker='X', c='brown',  label='adv')
    plt.ylim([-0.25213, 3.56749])
    plt.xlim([-0.94303, 1.99217])
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 3, 0, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')
    plt.tight_layout()
    plt.savefig('teaser_start3.png', dpi=300)

    plt.figure(i+3, (5, 5))
    plt.scatter(neighbors_pca_embeddings[:, 0], neighbors_pca_embeddings[:, 1], s=10, marker='o', c='blue', alpha=0.5, label='normal $k$-NN')
    plt.scatter(neighbors_pca_embeddings_adv[:, 0], neighbors_pca_embeddings_adv[:, 1], s=10, marker='v', c='red' , alpha=0.5, label='adv $k$-NN')
    plt.scatter(top_helpful_pca_embeddings[:, 0]    , top_helpful_pca_embeddings[:, 1]    , s=10, marker='s', c='blue', label='normal most helpful')
    plt.scatter(vis_embeddings[:, 0]    , vis_embeddings[:, 1]    , s=200, marker='*', c='black', label='normal')
    plt.scatter(vis_embeddings_adv[:, 0], vis_embeddings_adv[:, 1], s=200, marker='X', c='brown',  label='adv')
    plt.ylim([-0.25213, 3.56749])
    plt.xlim([-0.94303, 1.99217])
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [3, 4, 0, 1, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')
    plt.tight_layout()
    plt.savefig('teaser_start4.png', dpi=300)

    plt.figure(i+4, (5, 5))
    plt.scatter(neighbors_pca_embeddings[:, 0], neighbors_pca_embeddings[:, 1], s=10, marker='o', c='blue', alpha=0.5, label='normal $k$-NN')
    plt.scatter(neighbors_pca_embeddings_adv[:, 0], neighbors_pca_embeddings_adv[:, 1], s=10, marker='v', c='red' , alpha=0.5, label='adv $k$-NN')
    plt.scatter(top_helpful_pca_embeddings[:, 0]    , top_helpful_pca_embeddings[:, 1]    , s=10, marker='s', c='blue', label='normal most helpful')
    plt.scatter(top_helpful_pca_embeddings_adv[:, 0], top_helpful_pca_embeddings_adv[:, 1], s=10, marker='^', c='red' , label='adv most helpful')
    plt.scatter(vis_embeddings[:, 0]    , vis_embeddings[:, 1]    , s=200, marker='*', c='black', label='normal')
    plt.scatter(vis_embeddings_adv[:, 0], vis_embeddings_adv[:, 1], s=200, marker='X', c='brown',  label='adv')
    plt.ylim([-0.25213, 3.56749])
    plt.xlim([-0.94303, 1.99217])
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4, 5, 0, 1, 2, 3]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper left')
    plt.tight_layout()
    plt.savefig('teaser_start5.png', dpi=300)

