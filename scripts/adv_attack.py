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
from tqdm import tqdm

import darkon.darkon as darkon

from cleverhans.attacks import FastGradientMethod, DeepFool, SaliencyMapMethod, CarliniWagnerL2
from tensorflow_TB.cleverhans_alias.cw_attack_nnif import CarliniNNIF
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

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 100, 'Size of training batches')
flags.DEFINE_string('dataset', 'cifar10', 'datasset: cifar10/100 or svhn')
flags.DEFINE_string('set', 'val', 'val or test set to evaluate')
flags.DEFINE_string('attack', 'cw_nnif', 'adversarial attack: deepfool, jsma, cw, cw_nnif')

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
    NUM_INDICES = 50
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
    NUM_INDICES = 5
elif FLAGS.dataset == 'svhn':
    _classes = (
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    )
    ARCH_NAME = 'model_svhn'
    CHECKPOINT_NAME = 'svhn_mini/log_300519_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_exp1'
    LABEL_SMOOTHING = 0.1
    NUM_INDICES = 50
else:
    raise AssertionError('dataset {} not supported'.format(FLAGS.dataset))

weight_decay = 0.0004
TARGETED = FLAGS.attack != 'deepfool'

# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
superseed = 15101985
rand_gen = np.random.RandomState(superseed)
tf.set_random_seed(superseed)

# Set logging level to see debug information
set_log_level(logging.DEBUG)

# Create TF session
config_args = dict(allow_soft_placement=True)
sess = tf.Session(config=tf.ConfigProto(**config_args))

# get records from training
model_dir     = os.path.join('/data/gilad/logs/influence', CHECKPOINT_NAME)
workspace_dir = os.path.join(model_dir, WORKSPACE)
attack_dir    = os.path.join(model_dir, FLAGS.attack)
if TARGETED:
    attack_dir = attack_dir + '_targeted'

# make sure the attack dir is constructed
if not os.path.exists(attack_dir):
    os.makedirs(attack_dir)

mini_train_inds = None
if USE_TRAIN_MINI:
    print('loading train mini indices from {}'.format(os.path.join(model_dir, 'train_mini_indices.npy')))
    mini_train_inds = np.load(os.path.join(model_dir, 'train_mini_indices.npy'))

val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))
feeder = MyFeederValTest(dataset=FLAGS.dataset, rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                         test_val_set=test_val_set, mini_train_inds=mini_train_inds)

# get the data
X_train, y_train       = feeder.train_indices(range(feeder.get_train_size()))
X_val, y_val           = feeder.val_indices(range(feeder.get_val_size()))
X_test, y_test         = feeder.test_data, feeder.test_label  # getting the real test set
y_train_sparse         = y_train.argmax(axis=-1).astype(np.int32)
y_val_sparse           = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse          = y_test.argmax(axis=-1).astype(np.int32)

if TARGETED:
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

# Use Image Parameters
img_rows, img_cols, nchannels = X_test.shape[1:4]
nb_classes = y_test.shape[1]

# Define input TF placeholder
x     = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels), name='x')
y     = tf.placeholder(tf.float32, shape=(None, nb_classes), name='y')

eval_params = {'batch_size': FLAGS.batch_size}

model = DarkonReplica(scope=ARCH_NAME, nb_classes=feeder.num_classes, n=5, input_shape=[32, 32, 3])
preds      = model.get_predicted_class(x)
logits     = model.get_logits(x)
embeddings = model.get_embeddings(x)

loss = CrossEntropy(model, smoothing=LABEL_SMOOTHING)
regu_losses = WeightDecay(model)
full_loss = WeightedSum(model, [(1.0, loss), (weight_decay, regu_losses)])

def do_eval(preds, x_set, y_set, report_key, is_adv=None):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    setattr(report, report_key, acc)
    if is_adv is None:
        report_text = None
    elif is_adv:
        report_text = 'adversarial'
    else:
        report_text = 'legitimate'
    if report_text:
        print('Test accuracy on %s examples: %0.4f' % (report_text, acc))
    return acc

# loading the checkpoint
saver = tf.train.Saver()
checkpoint_path = os.path.join(model_dir, 'best_model.ckpt')
saver.restore(sess, checkpoint_path)

# predict labels from trainset
if USE_TRAIN_MINI:
    train_preds_file    = os.path.join(model_dir, 'x_train_mini_preds.npy')
    train_features_file = os.path.join(model_dir, 'x_train_mini_features.npy')
else:
    train_preds_file    = os.path.join(model_dir, 'x_train_preds.npy')
    train_features_file = os.path.join(model_dir, 'x_train_features.npy')
if not os.path.isfile(train_preds_file):
    x_train_preds, x_train_features = np_evaluate(sess, [preds, embeddings], X_train, y_train, x, y, FLAGS.batch_size, log=logging)
    x_train_preds = x_train_preds.astype(np.int32)
    np.save(train_preds_file, x_train_preds)
    np.save(train_features_file, x_train_features)
else:
    x_train_preds    = np.load(train_preds_file)
    x_train_features = np.load(train_features_file)

# predict labels from validation set
if not os.path.isfile(os.path.join(model_dir, 'x_val_preds.npy')):
    tf_inputs    = [x, y]
    tf_outputs   = [preds, embeddings]
    numpy_inputs = [X_val, y_val]

    x_val_preds, x_val_features = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_val_preds = x_val_preds.astype(np.int32)
    np.save(os.path.join(model_dir, 'x_val_preds.npy')   , x_val_preds)
    np.save(os.path.join(model_dir, 'x_val_features.npy'), x_val_features)
else:
    x_val_preds    = np.load(os.path.join(model_dir, 'x_val_preds.npy'))
    x_val_features = np.load(os.path.join(model_dir, 'x_val_features.npy'))

# predict labels from test set
if not os.path.isfile(os.path.join(model_dir, 'x_test_preds.npy')):
    tf_inputs    = [x, y]
    tf_outputs   = [preds, embeddings]
    numpy_inputs = [X_test, y_test]

    x_test_preds, x_test_features = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_test_preds = x_test_preds.astype(np.int32)
    np.save(os.path.join(model_dir, 'x_test_preds.npy')   , x_test_preds)
    np.save(os.path.join(model_dir, 'x_test_features.npy'), x_test_features)
else:
    x_test_preds    = np.load(os.path.join(model_dir, 'x_test_preds.npy'))
    x_test_features = np.load(os.path.join(model_dir, 'x_test_features.npy'))

# quick computations (without adv)
train_acc    = np.mean(y_train_sparse == x_train_preds)
val_acc      = np.mean(y_val_sparse   == x_val_preds)
test_acc     = np.mean(y_test_sparse  == x_test_preds)
print('train set acc: {}\nvalidation set acc: {}\ntest set acc: {}'.format(train_acc, val_acc, test_acc))

# what are the indices of the cifar10 set which the network succeeded classifying correctly,
# but the adversarial attack changed to a different class?
info_tmp = {}
info_tmp['val'] = {}
for i, set_ind in enumerate(feeder.val_inds):
    info_tmp['val'][i] = {}
    net_succ    = x_val_preds[i] == y_val_sparse[i]
    # attack_succ = x_val_preds[i] != x_val_preds_adv[i]  # the attack success is unknown yet
    info_tmp['val'][i]['global_index'] = set_ind
    info_tmp['val'][i]['net_succ']     = net_succ
    # info_tmp['val'][i]['attack_succ']  = attack_succ
info_tmp['test'] = {}
for i, set_ind in enumerate(feeder.test_inds):
    info_tmp['test'][i] = {}
    net_succ    = x_test_preds[i] == y_test_sparse[i]
    # attack_succ = x_test_preds[i] != x_test_preds_adv[i]
    info_tmp['test'][i]['global_index'] = set_ind
    info_tmp['test'][i]['net_succ']     = net_succ
    # info_tmp['test'][i]['attack_succ']  = attack_succ  # the attack success is unknown yet

# sub_relevant_indices = [ind for ind in info[FLAGS.set] if info[FLAGS.set][ind]['net_succ'] and info[FLAGS.set][ind]['attack_succ']]
# sub_relevant_indices = [ind for ind in info[FLAGS.set] if not info[FLAGS.set][ind]['attack_succ']]
sub_relevant_indices = [ind for ind in info_tmp[FLAGS.set]]
relevant_indices     = [info_tmp[FLAGS.set][ind]['global_index'] for ind in sub_relevant_indices]

helpful_npy_path = os.path.join(attack_dir, '{}_most_helpful.npy'.format(FLAGS.set))
harmful_npy_path = os.path.join(attack_dir, '{}_most_harmful.npy'.format(FLAGS.set))

if not os.path.exists(helpful_npy_path):
    # loading the embedding vectors of all the val's/test's most harmful/helpful training examples
    most_helpful_list = []
    most_harmful_list = []

    for i in tqdm(range(len(sub_relevant_indices))):
        # DEBUG:
        # if i >= 10:
        #     break
        sub_index = sub_relevant_indices[i]
        if test_val_set:
            global_index = feeder.val_inds[sub_index]
        else:
            global_index = feeder.test_inds[sub_index]
        assert global_index == relevant_indices[i]

        _, real_label = feeder.test_indices(sub_index)
        real_label = np.argmax(real_label)

        if test_val_set:
            pred_label = x_val_preds[sub_index]
        else:
            pred_label = x_test_preds[sub_index]

        if info_tmp[FLAGS.set][sub_index]['net_succ']:
            assert pred_label == real_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)

        progress_str = 'sample {}/{}: processing helpful/harmful for {} index {} (sub={}).\n' \
                       'real label: {}, pred label: {}. net_succ={}' \
            .format(i + 1, len(sub_relevant_indices), FLAGS.set, global_index, sub_index, _classes[real_label],
                    _classes[pred_label], info_tmp[FLAGS.set][sub_index]['net_succ'])
        logging.info(progress_str)
        print(progress_str)

        if not info_tmp[FLAGS.set][sub_index]['net_succ']:  # if prediction is different than real
            case = 'pred'
        else:
            case = 'real'

        # creating the relevant index folders
        dir = os.path.join(model_dir, FLAGS.set, FLAGS.set + '_index_{}'.format(global_index), case)
        scores = np.load(os.path.join(dir, 'scores.npy'))
        sorted_indices = np.argsort(scores)
        harmful_inds = sorted_indices[:NUM_INDICES]
        helpful_inds = sorted_indices[-NUM_INDICES:][::-1]

        # find out the embedding space of the train images in the tanh space
        # first we calculate the tanh transformation:
        X_train_transform = (np.tanh(X_train) + 1) / 2

        most_helpful_images = X_train_transform[helpful_inds]
        most_harmful_images = X_train_transform[harmful_inds]
        train_helpful_embeddings = batch_eval(sess, [x, y], [embeddings], [most_helpful_images, y_train[helpful_inds]], FLAGS.batch_size)[0]
        train_harmful_embeddings = batch_eval(sess, [x, y], [embeddings], [most_harmful_images, y_train[harmful_inds]], FLAGS.batch_size)[0]

        most_helpful_list.append(train_helpful_embeddings)
        most_harmful_list.append(train_harmful_embeddings)

    most_helpful = np.asarray(most_helpful_list)
    most_harmful = np.asarray(most_harmful_list)
    np.save(helpful_npy_path, most_helpful)
    np.save(harmful_npy_path, most_harmful)
else:
    print('{} already exist. Loading...'.format(helpful_npy_path))
    most_helpful = np.load(helpful_npy_path)
    most_harmful = np.load(harmful_npy_path)

# DEBUG:
# most_helpful = np.tile(most_helpful, [100, 1, 1])
# most_harmful = np.tile(most_harmful, [100, 1, 1])

# initialize adversarial examples if necessary
if not os.path.exists(os.path.join(attack_dir, 'X_{}_adv.npy'.format(FLAGS.set))):
    y_adv     = tf.placeholder(tf.float32, shape=(None, nb_classes), name='y_adv')
    m_help_ph = tf.placeholder(tf.float32, shape=(None,) + most_helpful.shape[1:])
    m_harm_ph = tf.placeholder(tf.float32, shape=(None,) + most_harmful.shape[1:])

    # Initialize the advarsarial attack object and graph
    attack_params = {
        'clip_min': 0.0,
        'clip_max': 1.0,
        'batch_size': FLAGS.batch_size,
        'confidence': 0.8,
        'learning_rate': 0.01,
        'initial_const': 0.1,
        'y_target': y_adv,
        'most_helpful_locs': m_help_ph,
        'most_harmful_locs': m_harm_ph
    }
    attack_class = CarliniNNIF

    attack         = attack_class(model, sess=sess)
    adv_x          = attack.generate(x, **attack_params)
    preds_adv      = model.get_predicted_class(adv_x)
    logits_adv     = model.get_logits(adv_x)
    embeddings_adv = model.get_embeddings(adv_x)

    # attack
    tf_inputs    = [x, y, y_adv, m_help_ph, m_harm_ph]
    tf_outputs   = [adv_x, preds_adv, embeddings_adv]
    if FLAGS.set == 'val':
        numpy_inputs = [X_val, y_val, y_val_targets, most_helpful, most_harmful]
    elif FLAGS.set == 'test':
        numpy_inputs = [X_test, y_test, y_test_targets, most_helpful, most_harmful]

    X_set_adv, x_set_preds_adv, x_set_features_adv = batch_eval(sess, tf_inputs, tf_outputs, numpy_inputs, FLAGS.batch_size)
    x_set_preds_adv = x_set_preds_adv.astype(np.int32)
    np.save(os.path.join(attack_dir, 'X_{}_adv.npy'.format(FLAGS.set))         , X_set_adv)
    np.save(os.path.join(attack_dir, 'x_{}_preds_adv.npy'.format(FLAGS.set))   , x_set_preds_adv)
    np.save(os.path.join(attack_dir, 'x_{}_features_adv.npy'.format(FLAGS.set)), x_set_features_adv)
else:
    print('{} already exists'.format(os.path.join(attack_dir, 'X_{}_adv.npy'.format(FLAGS.set))))
