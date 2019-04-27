from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import os

from threading import Thread
from Queue import Queue

import darkon.darkon as darkon

from cleverhans.attacks import FastGradientMethod, DeepFool
from tensorflow.python.platform import flags
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
from cleverhans.loss import CrossEntropy, WeightDecay, WeightedSum
from tensorflow_TB.lib.models.darkon_replica_model import DarkonReplica
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from tensorflow_TB.utils.misc import one_hot
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tensorflow_TB.lib.datasets.influence_feeder_val_test import MyFeederValTest
from tensorflow_TB.utils.misc import np_evaluate
import copy
import pickle
import imageio

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_float('weight_decay', 0.0004, 'weight decay')
flags.DEFINE_string('checkpoint_name', 'log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000', 'checkpoint name')
flags.DEFINE_float('label_smoothing', 0.1, 'label smoothing')
flags.DEFINE_string('workspace', 'influence_workspace_test_mini', 'workspace dir')
flags.DEFINE_bool('prepare', False, 'whether or not we are in the prepare phase, when hvp is calculated')
flags.DEFINE_string('set', 'test', 'val or test set to evaluate')
flags.DEFINE_bool('use_train_mini', True, 'Whether or not to use 5000 training samples instead of 49000')
flags.DEFINE_integer('num_threads', 20, 'number of threads')

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

# Get CIFAR-10 data
cifar10_input.maybe_download_and_extract()

# get records from training
model_dir     = os.path.join('/data/gilad/logs/influence', FLAGS.checkpoint_name)
workspace_dir = os.path.join(model_dir, FLAGS.workspace)

mini_train_inds = None
if FLAGS.use_train_mini:
    print('loading train mini indices from {}'.format(os.path.join(model_dir, 'train_mini_indices.npy')))
    mini_train_inds = np.load(os.path.join(model_dir, 'train_mini_indices.npy'))

save_val_inds = False
if os.path.isfile(os.path.join(model_dir, 'val_indices.npy')):
    print('re-using val indices from {}'.format(os.path.join(model_dir, 'val_indices.npy')))
    val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))
else:
    val_indices = None
    save_val_inds = True
feeder = MyFeederValTest(rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices,
                         test_val_set=test_val_set, mini_train_inds=mini_train_inds)
if save_val_inds:
    print('saving new val indices to'.format(os.path.join(model_dir, 'val_indices.npy')))
    np.save(os.path.join(model_dir, 'val_indices.npy'), feeder.val_inds)

# get the data
X_train, y_train       = feeder.train_indices(range(feeder.get_train_size()))
X_val, y_val           = feeder.val_indices(range(feeder.get_val_size()))
X_test, y_test         = feeder.test_data, feeder.test_label  # getting the real test set
y_train_sparse         = y_train.argmax(axis=-1).astype(np.int32)
y_val_sparse           = y_val.argmax(axis=-1).astype(np.int32)
y_test_sparse          = y_test.argmax(axis=-1).astype(np.int32)

# Use Image Parameters
img_rows, img_cols, nchannels = X_test.shape[1:4]
nb_classes = y_test.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

eval_params = {'batch_size': FLAGS.batch_size}
fgsm_params = {
    'eps': 0.3,
    'clip_min': 0.,
    'clip_max': 1.
}
deepfool_params = {
    'clip_min': 0.0,
    'clip_max': 1.0
}

model = DarkonReplica(scope='model1', nb_classes=10, n=5, input_shape=[32, 32, 3])
preds      = model.get_predicted_class(x)
logits     = model.get_logits(x)
embeddings = model.get_embeddings(x)

loss = CrossEntropy(model, smoothing=FLAGS.label_smoothing)
regu_losses = WeightDecay(model)
full_loss = WeightedSum(model, [(1.0, loss), (FLAGS.weight_decay, regu_losses)])

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
if FLAGS.use_train_mini:
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
    x_val_preds, x_val_features = np_evaluate(sess, [preds, embeddings], X_val, y_val, x, y, FLAGS.batch_size, log=logging)
    x_val_preds = x_val_preds.astype(np.int32)
    np.save(os.path.join(model_dir, 'x_val_preds.npy'), x_val_preds)
    np.save(os.path.join(model_dir, 'x_val_features.npy'), x_val_features)
else:
    x_val_preds    = np.load(os.path.join(model_dir, 'x_val_preds.npy'))
    x_val_features = np.load(os.path.join(model_dir, 'x_val_features.npy'))

# predict labels from test set
if not os.path.isfile(os.path.join(model_dir, 'x_test_preds.npy')):
    x_test_preds, x_test_features = np_evaluate(sess, [preds, embeddings], X_test, y_test, x, y, FLAGS.batch_size, log=logging)
    x_test_preds = x_test_preds.astype(np.int32)
    np.save(os.path.join(model_dir, 'x_test_preds.npy'), x_test_preds)
    np.save(os.path.join(model_dir, 'x_test_features.npy'), x_test_features)
else:
    x_test_preds    = np.load(os.path.join(model_dir, 'x_test_preds.npy'))
    x_test_features = np.load(os.path.join(model_dir, 'x_test_features.npy'))

# Initialize the advarsarial attack object and graph
attack         = DeepFool(model, sess=sess)
adv_x          = attack.generate(x, **deepfool_params)
preds_adv      = model.get_predicted_class(adv_x)
logits_adv     = model.get_logits(adv_x)
embeddings_adv = model.get_embeddings(adv_x)

if not os.path.isfile(os.path.join(model_dir, 'X_val_adv.npy')):
    # Evaluate the accuracy of the CIFAR-10 model on adversarial examples
    X_val_adv, x_val_preds_adv, x_val_features_adv = np_evaluate(sess, [adv_x, preds_adv, embeddings_adv], X_val, y_val, x, y, FLAGS.batch_size, log=logging)
    x_val_preds_adv = x_val_preds_adv.astype(np.int32)
    # since DeepFool is not reproducible, saving the results in as numpy
    np.save(os.path.join(model_dir, 'X_val_adv.npy'), X_val_adv)
    np.save(os.path.join(model_dir, 'x_val_preds_adv.npy'), x_val_preds_adv)
    np.save(os.path.join(model_dir, 'x_val_features_adv.npy'), x_val_features_adv)
else:
    X_val_adv          = np.load(os.path.join(model_dir, 'X_val_adv.npy'))
    x_val_preds_adv    = np.load(os.path.join(model_dir, 'x_val_preds_adv.npy'))
    x_val_features_adv = np.load(os.path.join(model_dir, 'x_val_features_adv.npy'))

if not os.path.isfile(os.path.join(model_dir, 'X_test_adv.npy')):
    # Evaluate the accuracy of the CIFAR-10 model on adversarial examples
    X_test_adv, x_test_preds_adv, x_test_features_adv = np_evaluate(sess, [adv_x, preds_adv, embeddings_adv], X_test, y_test, x, y, FLAGS.batch_size, log=logging)
    x_test_preds_adv = x_test_preds_adv.astype(np.int32)
    # since DeepFool is not reproducible, saving the results in as numpy
    np.save(os.path.join(model_dir, 'X_test_adv.npy'), X_test_adv)
    np.save(os.path.join(model_dir, 'x_test_preds_adv.npy'), x_test_preds_adv)
    np.save(os.path.join(model_dir, 'x_test_features_adv.npy'), x_test_features_adv)
else:
    X_test_adv          = np.load(os.path.join(model_dir, 'X_test_adv.npy'))
    x_test_preds_adv    = np.load(os.path.join(model_dir, 'x_test_preds_adv.npy'))
    x_test_features_adv = np.load(os.path.join(model_dir, 'x_test_features_adv.npy'))

# accuracy computation
# do_eval(logits, X_train, y_train, 'clean_train_clean_eval_trainset', False)
# do_eval(logits, X_val, y_val, 'clean_train_clean_eval_validationset', False)
# do_eval(logits, X_test, y_test, 'clean_train_clean_eval_testset', False)
# do_eval(logits_adv, X_val, y_val, 'clean_train_adv_eval_validationset', True)
# do_eval(logits_adv, X_test, y_test, 'clean_train_adv_eval_testset', True)

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

# start the knn observation
knn = NearestNeighbors(n_neighbors=feeder.get_train_size(), p=2, n_jobs=20)
knn.fit(x_train_features)
if test_val_set:
    print('predicting knn for all val set')
    features = x_val_features
else:
    print('predicting knn for all test set')
    features = x_test_features
all_neighbor_dists, all_neighbor_indices = knn.kneighbors(features, return_distance=True)

# setting pred feeder
pred_feeder = MyFeederValTest(rand_gen=rand_gen, as_one_hot=True,
                              val_inds=feeder.val_inds, test_val_set=test_val_set, mini_train_inds=mini_train_inds)
pred_feeder.val_origin_data  = X_val_adv
pred_feeder.val_data         = X_val_adv
pred_feeder.val_label        = one_hot(x_val_preds, 10).astype(np.float32)
pred_feeder.test_origin_data = X_test_adv
pred_feeder.test_data        = X_test_adv
pred_feeder.test_label       = one_hot(x_test_preds, 10).astype(np.float32)

# setting adv feeder
adv_feeder = MyFeederValTest(rand_gen=rand_gen, as_one_hot=True,
                             val_inds=feeder.val_inds, test_val_set=test_val_set, mini_train_inds=mini_train_inds)
adv_feeder.val_origin_data  = X_val_adv
adv_feeder.val_data         = X_val_adv
adv_feeder.val_label        = one_hot(x_val_preds_adv, 10).astype(np.float32)
adv_feeder.test_origin_data = X_test_adv
adv_feeder.test_data        = X_test_adv
adv_feeder.test_label       = one_hot(x_test_preds_adv, 10).astype(np.float32)

# now finding the influence
feeder.reset()
pred_feeder.reset()
adv_feeder.reset()

inspector_list = []
inspector_pred_list = []
inspector_adv_list = []

for ii in range(FLAGS.num_threads):
    inspector_list.append(
        darkon.Influence(
            workspace=os.path.join(model_dir, FLAGS.workspace, 'real'),
            feeder=copy.deepcopy(feeder),
            loss_op_train=full_loss.fprop(x=x, y=y),
            loss_op_test=loss.fprop(x=x, y=y),
            x_placeholder=x,
            y_placeholder=y)
    )
    inspector_pred_list.append(
        darkon.Influence(
            workspace=os.path.join(model_dir, FLAGS.workspace, 'pred'),
            feeder=copy.deepcopy(pred_feeder),
            loss_op_train=full_loss.fprop(x=x, y=y),
            loss_op_test=loss.fprop(x=x, y=y),
            x_placeholder=x,
            y_placeholder=y)
    )
    inspector_adv_list.append(
        darkon.Influence(
            workspace=os.path.join(model_dir, FLAGS.workspace, 'adv'),
            feeder=copy.deepcopy(adv_feeder),
            loss_op_train=full_loss.fprop(x=x, y=y),
            loss_op_test=loss.fprop(x=x, y=y),
            x_placeholder=x,
            y_placeholder=y)
    )

testset_batch_size = 100
train_batch_size = 100
train_iterations = 50 if FLAGS.use_train_mini else 490  # was 500 wo validation

approx_params = {
    'scale': 200,
    'num_repeats': 5,
    'recursion_depth': 10 if FLAGS.use_train_mini else 98,  # 5000 for mini and 49000 for regular train
    'recursion_batch_size': 100
}

# sub_relevant_indices = [ind for ind in info[FLAGS.set] if info[FLAGS.set][ind]['net_succ'] and info[FLAGS.set][ind]['attack_succ']]
sub_relevant_indices = [ind for ind in info[FLAGS.set]]
relevant_indices     = [info[FLAGS.set][ind]['global_index'] for ind in sub_relevant_indices]

# b, e = 7056, 10000
# sub_relevant_indices = sub_relevant_indices[b:e]
# relevant_indices     = relevant_indices[b:e]

def collect_influence(q, thread_id):
    while not q.empty():
        work = q.get()
        try:
            i = work[0]
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

            _, adv_label = adv_feeder.test_indices(sub_index)
            adv_label = np.argmax(adv_label)

            if info[FLAGS.set][sub_index]['attack_succ']:
                assert pred_label != adv_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)
            if info[FLAGS.set][sub_index]['net_succ']:
                assert pred_label == real_label, 'failed for i={}, sub_index={}, global_index={}'.format(i, sub_index, global_index)
            progress_str = 'thread_id: {}. sample {}/{}: calculating scores for {} index {} (sub={}).\n' \
                           'real label: {}, adv label: {}, pred label: {}. net_succ={}, attack_succ={}' \
                .format(thread_id, i + 1, len(sub_relevant_indices), FLAGS.set, global_index, sub_index, _classes[real_label],
                        _classes[adv_label], _classes[pred_label], info[FLAGS.set][sub_index]['net_succ'], info[FLAGS.set][sub_index]['attack_succ'])
            logging.info(progress_str)
            print(progress_str)

            cases = ['real']
            if not info[FLAGS.set][sub_index]['net_succ']:  # if prediction is different than real
                cases.append('pred')
            if info[FLAGS.set][sub_index]['attack_succ']:  # if adv is different than prediction
                cases.append('adv')
            for case in cases:
                if case == 'real':
                    feed = feeder
                    insp = inspector_list[thread_id]
                elif case == 'pred':
                    feed = pred_feeder
                    insp = inspector_pred_list[thread_id]
                elif case == 'adv':
                    feed = adv_feeder
                    insp = inspector_adv_list[thread_id]
                else:
                    raise AssertionError('only real and adv are accepted.')

                if FLAGS.prepare:
                    insp._prepare(
                        sess=sess,
                        test_indices=[sub_index],
                        test_batch_size=testset_batch_size,
                        approx_params=approx_params,
                        force_refresh=False
                    )
                else:
                    # creating the relevant index folders
                    dir = os.path.join(model_dir, FLAGS.set, FLAGS.set + '_index_{}'.format(global_index), case)
                    if not os.path.exists(dir):
                        os.makedirs(dir)

                    if os.path.isfile(os.path.join(dir, 'scores.npy')):
                        print('loading scores from {}'.format(os.path.join(dir, 'scores.npy')))
                        scores = np.load(os.path.join(dir, 'scores.npy'))
                    else:
                        scores = insp.upweighting_influence_batch(
                            sess=sess,
                            test_indices=[sub_index],
                            test_batch_size=testset_batch_size,
                            approx_params=approx_params,
                            train_batch_size=train_batch_size,
                            train_iterations=train_iterations)
                        np.save(os.path.join(dir, 'scores.npy'), scores)

                    if not os.path.isfile(os.path.join(dir, 'image.png')):
                        print('saving image to {}'.format(os.path.join(dir, 'image.npy/png')))
                        image, _ = feed.test_indices(sub_index)
                        imageio.imwrite(os.path.join(dir, 'image.png'), image)
                        np.save(os.path.join(dir, 'image.npy'), image)
                    else:
                        # verifying everything is good
                        assert (np.load(os.path.join(dir, 'image.npy')) == feed.test_indices(sub_index)[0]).all()
        except Exception as e:
            print('Error with influence collect function for i={}: {}'.format(i, e))
            exit(1)
            raise AssertionError('Error with influence collect function for i={}!'.format(i))

        # signal to the queue that task has been processed
        q.task_done()
    return True

# set up a queue to hold all the jobs:
q = Queue(maxsize=0)
# for i in range(len(sub_relevant_indices)):
for i in range(7056, 10000):
    q.put((i,))

for thread_id in range(FLAGS.num_threads):
    print('Starting thread {}'.format(thread_id))
    worker = Thread(target=collect_influence, args=(q, thread_id))
    worker.setDaemon(True)
    worker.start()

q.join()
print('All tasks completed.')
