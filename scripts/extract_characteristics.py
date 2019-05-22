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
from tqdm import tqdm
import sklearn.covariance

from lid_adversarial_subspace_detection.util import mle_batch

# tf.enable_eager_execution()

STDEVS = {
    'cifar10': {'deepfool': 0.00796, 'cw': 0.007}
}

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_string('dataset', 'cifar10', 'dataset: cifar10/100 or svhn')
flags.DEFINE_string('attack', 'deepfool', 'adversarial attack: deepfool, jsma, cw')
flags.DEFINE_bool('targeted', False, 'whether or not the adversarial attack is targeted')
flags.DEFINE_string('characteristics', 'mahalanobis', 'type of defence')
flags.DEFINE_integer('k_nearest', 100, 'number of nearest neighbors to use for LID detection')
flags.DEFINE_float('magnitude', 0.002, 'magnitude for mahalanobis detection')

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
model_dir          = os.path.join('/data/gilad/logs/influence', CHECKPOINT_NAME)
attack_dir         = os.path.join(model_dir, FLAGS.attack)
if FLAGS.targeted:
    attack_dir = attack_dir + '_targeted'
characteristics_dir = os.path.join(attack_dir, FLAGS.characteristics)
if not os.path.exists(characteristics_dir):
    os.makedirs(characteristics_dir)

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

# DEBUG: testing different scale so that L2 perturbation is the same
# diff_adv    = X_test_adv.reshape((len(X_test), -1)) - X_test.reshape((len(X_test), -1))
# l2_diff_adv = np.linalg.norm(diff_adv, axis=1).mean()
# for std in np.arange(0.0079, 0.008, 0.00001):
#     X_test_noisy = get_noisy_samples(X_test, X_test_adv, std)
#     diff = X_test_noisy.reshape((len(X_test), -1)) - X_test.reshape((len(X_test), -1))
#     l2_diff = np.linalg.norm(diff, axis=1).mean()
#     print('for std={}: diff of L2 perturbations is {}'.format(std, l2_diff - l2_diff_adv))

noisy_file = os.path.join(attack_dir, 'X_test_noisy.npy')
if os.path.isfile(noisy_file):
    print('Loading {} noisy samples from {}'.format(FLAGS.dataset, noisy_file))
    X_test_noisy = np.load(noisy_file)
else:
    print('Crafting {} noisy samples.'.format(FLAGS.dataset))
    X_test_noisy = get_noisy_samples(X_test, X_test_adv)
    np.save(noisy_file, X_test_noisy)

for s_type, subset in zip(['normal', 'noisy', 'adversarial'], [X_test, X_test_noisy, X_test_adv]):
    # acc = model_eval(sess, x, y, logits, subset, y_test, args=eval_params)
    # print("Model accuracy on the %s test set: %0.2f%%" % (s_type, 100 * acc))
    # Compute and display average perturbation sizes
    if not s_type == 'normal':
        diff    = subset.reshape((len(X_test), -1)) - X_test.reshape((len(X_test), -1))
        l2_diff = np.linalg.norm(diff, axis=1).mean()
        print("Average L-2 perturbation size of the %s test set: %0.2f" % (s_type, l2_diff))

# Refine the normal, noisy and adversarial sets to only include samples for
# which the original version was correctly classified by the model
inds_correct = np.where(x_test_preds == y_test_sparse)[0]
print("Number of correctly predict images: %s" % (len(inds_correct)))

X_test              = X_test[inds_correct]
X_test_noisy        = X_test_noisy[inds_correct]
X_test_adv          = X_test_adv[inds_correct]
x_test_preds        = x_test_preds[inds_correct]
x_test_features     = x_test_features[inds_correct]
x_test_preds_adv    = x_test_preds_adv[inds_correct]
x_test_features_adv = x_test_features_adv[inds_correct]

y_test              = y_test[inds_correct]
y_test_sparse       = y_test_sparse[inds_correct]
print("X_test: ", X_test.shape)
print("X_test_noisy: ", X_test_noisy.shape)
print("X_test_adv: ", X_test_adv.shape)

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def get_lids_random_batch(X_test, X_test_noisy, X_test_adv, k=FLAGS.k_nearest, batch_size=100):
    """
    :param X_test: normal images
    :param X_test_noisy: noisy images
    :param X_test_adv: advserial images
    :param k: the number of nearest neighbours for LID estimation
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    lid_dim = 1  # just taking the embedding space

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X_test), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch       = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv   = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_noisy = np.zeros(shape=(n_feed, lid_dim))

        # applying only on the embedding space
        X_act       = batch_eval(sess, [x], [embeddings], [X_test[start:end]], batch_size)[0]
        X_act       = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
        X_adv_act   = batch_eval(sess, [x], [embeddings], [X_test_adv[start:end]], batch_size)[0]
        X_adv_act   = np.asarray(X_adv_act, dtype=np.float32).reshape((n_feed, -1))
        X_noisy_act = batch_eval(sess, [x], [embeddings], [X_test_noisy[start:end]], batch_size)[0]
        X_noisy_act = np.asarray(X_noisy_act, dtype=np.float32).reshape((n_feed, -1))

        # random clean samples
        # Maximum likelihood estimation of local intrinsic dimensionality (LID)
        lid_batch[:, 0]       = mle_batch(X_act, X_act      , k=k)
        lid_batch_adv[:, 0]   = mle_batch(X_act, X_adv_act  , k=k)
        lid_batch_noisy[:, 0] = mle_batch(X_act, X_noisy_act, k=k)

        return lid_batch, lid_batch_noisy, lid_batch_adv

    lids = []
    lids_adv = []
    lids_noisy = []
    n_batches = int(np.ceil(X_test.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_noisy, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
        lids_noisy.extend(lid_batch_noisy)

    lids       = np.asarray(lids, dtype=np.float32)
    lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
    lids_adv   = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_noisy, lids_adv

def get_lid(X_test, X_test_noisy, X_test_adv, k=FLAGS.k_nearest, batch_size=100):
    print('Extract local intrinsic dimensionality: k = %s' % FLAGS.k_nearest)
    lids_normal, lids_noisy, lids_adv = get_lids_random_batch(X_test, X_test_noisy, X_test_adv, k, batch_size)
    print("lids_normal:", lids_normal.shape)
    print("lids_noisy:", lids_noisy.shape)
    print("lids_adv:", lids_adv.shape)

    lids_pos = lids_adv
    lids_neg = np.concatenate((lids_normal, lids_noisy))
    artifacts, labels = merge_and_generate_labels(lids_pos, lids_neg)

    return artifacts, labels

def sample_estimator(num_classes, X, Y, x_preds, x_features):
    num_output = 1  # can be different in the future
    num_sample_per_class = np.zeros(num_classes)
    list_features = []
    num_feature = 64

    accuracy = np.mean(x_preds == Y)

    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    out_count = 0
    for label in Y:
        for out_id in range(num_output):  # maybe do in the future
            list_features[out_id][label] = x_features[Y == label]
        num_sample_per_class[label] = list_features[out_count][label].shape[0]

    sample_class_mean = []
    for k in range(num_output):  # maybe do in the future
        temp_list = np.zeros((num_classes, num_feature))
        for i in range(num_classes):
            temp_list[i] = np.mean(list_features[out_count][i], 0)
        sample_class_mean.append(temp_list)

    precision = []
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    for k in range(num_output):
        D = 0
        for i in range(num_classes):
            if i == 0:
                D = list_features[k][i] - sample_class_mean[k][i]
            else:
                D = np.concatenate((D, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        group_lasso.fit(D)
        temp_precision = group_lasso.precision_
        precision.append(temp_precision)

    print('\n Training Accuracy: {}%)\n'.format(accuracy * 100.0))

    return sample_class_mean, precision

def get_Mahalanobis_score_adv(test_data, gaussian_score, grads, magnitude, set):
    batch_size = 100
    grad_file = os.path.join(characteristics_dir, 'gradients_{}.npy'.format(set))
    if os.path.exists(grad_file):
        print('loading gradients from {}'.format(grad_file))
        gradients = np.load(grad_file)
    else:
        print('Calculating Mahanalobis gradients...')
        gradients = batch_eval(sess, [x], grads, [test_data], batch_size)[0]
        print('Saving gradients to {}'.format(grad_file))
        np.save(grad_file, gradients)

    gradients = gradients.clip(min=0)
    gradients = (gradients - 0.5) * 2

    # scale hyper params given from the official deep_Mahalanobis_detector repo:
    RED_SCALE   = 0.2023
    GREEN_SCALE = 0.1994
    BLUE_SCALE  = 0.2010

    gradients_scaled = np.zeros_like(gradients)
    gradients_scaled[:, :, :, 0] = gradients[:, :, :, 0] / RED_SCALE
    gradients_scaled[:, :, :, 1] = gradients[:, :, :, 1] / GREEN_SCALE
    gradients_scaled[:, :, :, 2] = gradients[:, :, :, 2] / BLUE_SCALE

    tempInputs = X_test - magnitude * gradients
    print('Calculating noise gaussian scores...')
    noise_gaussian_score = batch_eval(sess, [x], [gaussian_score], [tempInputs], batch_size)[0]

    Mahalanobis = np.max(noise_gaussian_score, axis=1)

    return Mahalanobis

def get_mahanabolis_tensors(sample_mean, precision, num_classes, layer_index=0):
    # here I try to calculte the input gradients for -pure_tau. Meaning d(-pure_tau)/dx.
    # First, how do we calculate pure_tau? This is a computation on a batch.
    precision_mat = tf.convert_to_tensor(precision[layer_index], dtype=tf.float32)
    sample_mean_tensor = tf.convert_to_tensor(sample_mean[layer_index], dtype=tf.float32)
    with tf.name_scope('Mahanabolis_grad_calc'):
        for i in range(num_classes):
            batch_sample_mean = sample_mean_tensor[i]
            zero_f = embeddings - batch_sample_mean
            zero_f_T = tf.transpose(zero_f)
            term_gau = -0.5 * tf.matmul(tf.matmul(zero_f, precision_mat), zero_f_T)
            term_gau = tf.diag_part(term_gau)
            if i == 0:
                gaussian_score = tf.reshape(term_gau, (-1, 1))
            else:
                gaussian_score_tmp = tf.reshape(term_gau, (-1, 1))
                gaussian_score = tf.concat([gaussian_score, gaussian_score_tmp], axis=1)

        # Input_processing
        sample_pred = tf.argmax(gaussian_score, axis=1)
        batch_sample_mean = tf.gather(sample_mean_tensor, axis=0, indices=sample_pred)
        zero_f = embeddings - tf.identity(batch_sample_mean)
        zero_f_T = tf.transpose(zero_f)
        pure_gau = -0.5 * tf.matmul(tf.matmul(zero_f, tf.identity(precision_mat)), zero_f_T)  # 100x100
        pure_gau = tf.diag_part(pure_gau)  # 100
        gau_loss = tf.reduce_mean(-pure_gau)
        grads = tf.gradients(gau_loss, x)

    return gaussian_score, grads

if FLAGS.characteristics == 'lid':
    characteristics, labels = get_lid(X_test, X_test_noisy, X_test_adv, FLAGS.k_nearest, 100)
    print("LID: [characteristic shape: ", characteristics.shape, ", label shape: ", labels.shape)
    file_name = os.path.join(characteristics_dir, 'k_{}_batch_{}.npy'.format(FLAGS.k_nearest, 100))
    data = np.concatenate((characteristics, labels), axis=1)
    np.save(file_name, data)

if FLAGS.characteristics == 'mahalanobis':
    print('get sample mean and covariance')
    # sample_mean[0].shape=(10,64)
    # precision[0].shape=(64,64)
    sample_mean, precision = sample_estimator(feeder.num_classes, X_train, y_train_sparse, x_train_preds, x_train_features)
    gaussian_score, grads  = get_mahanabolis_tensors(sample_mean, precision, feeder.num_classes)

    M_in    = get_Mahalanobis_score_adv(X_test      , gaussian_score, grads, FLAGS.magnitude, set='normal')
    M_out   = get_Mahalanobis_score_adv(X_test_adv  , gaussian_score, grads, FLAGS.magnitude, set='adv')
    M_noisy = get_Mahalanobis_score_adv(X_test_noisy, gaussian_score, grads, FLAGS.magnitude, set='noisy')

    Mahalanobis_neg = np.concatenate((M_in, M_noisy))
    Mahalanobis_pos = M_out
    characteristics, labels = merge_and_generate_labels(Mahalanobis_pos, Mahalanobis_neg)
    file_name = os.path.join(characteristics_dir, 'magnitude_{}.npy'.format(FLAGS.magnitude))
    data = np.concatenate((characteristics, labels), axis=1)
    np.save(file_name, data)


    # out_features = batch_eval(sess, [x], [embeddings], [test_data[start:end]], batch_size)[0]
# out_features = np.asarray(out_features, dtype=np.float32).reshape((n_feed, out_features.shape[1], -1))
# out_features = np.mean(out_features, axis=2)
#
# gaussian_score = 0
# for i in range(num_classes):
#     batch_sample_mean = sample_mean[layer_index][i]
#     zero_f = out_features - batch_sample_mean
#     term_gau = -0.5 * np.matmul(np.matmul(zero_f, precision[layer_index]), zero_f.T).diagonal()  #(100x64)x(64x64)x(64x100)
#     if i == 0:
#         gaussian_score = np.reshape(term_gau, (-1, 1))
#     else:
#         gaussian_score = np.concatenate((gaussian_score, np.reshape(term_gau, (-1, 1))), 1)
#
# # Input_processing
# # move gaussian score and many other to be pytorch tensors
# gaussian_score_tensor = torch.tensor(gaussian_score)
# sample_mean_tensor    = torch.tensor(sample_mean[layer_index])  # shape: (10x64)
# precision_tensor      = torch.tensor(precision[layer_index])  # shape: (64x64)
# out_features_tensor   = torch.tensor(out_features)  # shape: (100,64)
#
# # sample_pred = gaussian_score.argmax(axis=1)  # from (100x10) to (100,), taking the max
# sample_pred = gaussian_score_tensor.max(1)[1]  # shape:(100,)
# batch_sample_mean = sample_mean_tensor.index_select(0, sample_pred)  # shape:(100x64)
# zero_f = out_features_tensor.double() - Variable(batch_sample_mean)
# pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision_tensor)), zero_f.t()).diag()
# loss = torch.mean(-pure_gau)
# loss.backward()
