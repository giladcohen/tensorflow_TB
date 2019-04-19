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

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_float('weight_decay', 0.0004, 'weight decay')
flags.DEFINE_string('checkpoint_name', 'log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000', 'checkpoint name')
flags.DEFINE_float('label_smoothing', 0.1, 'label smoothing')
flags.DEFINE_string('workspace', 'influence_workspace_validation', 'workspace dir')
flags.DEFINE_bool('prepare', False, 'whether or not we are in the prepare phase, when hvp is calculated')
flags.DEFINE_integer('num_threads', 20, 'Size of training batches')


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

save_val_inds = False
if os.path.isfile(os.path.join(model_dir, 'val_indices.npy')):
    print('re-using val indices from {}'.format(os.path.join(model_dir, 'val_indices.npy')))
    val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))
else:
    val_indices = None
    save_val_inds = True
feeder = MyFeederValTest(rand_gen=rand_gen, as_one_hot=True, val_inds=val_indices, test_val_set=True)
if save_val_inds:
    print('saving new val indices to'.format(os.path.join(model_dir, 'val_indices.npy')))
    np.save(os.path.join(model_dir, 'val_indices.npy'), feeder.val_inds)

# get the data
X_train, y_train       = feeder.train_indices(range(49000))
X_val, y_val           = feeder.val_indices(range(1000))
X_test, y_test         = feeder.test_indices(range(1000))  # for the validation testing
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
x_train_preds, x_train_features = np_evaluate(sess, [preds, embeddings], X_train, y_train, x, y, FLAGS.batch_size, log=logging)
x_train_preds = x_train_preds.astype(np.int32)
#do_eval(logits, X_train, y_train, 'clean_train_clean_eval_trainset', False)
# predict labels from validation set
x_val_preds, x_val_features = np_evaluate(sess, [preds, embeddings], X_val, y_val, x, y, FLAGS.batch_size, log=logging)
x_val_preds = x_val_preds.astype(np.int32)
do_eval(logits, X_val, y_val, 'clean_train_clean_eval_validationset', False)
#np.mean(y_test.argmax(axis=-1) == x_test_preds)

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
    # do_eval(logits_adv, X_val, y_val, 'clean_train_adv_eval', True)
    # do quicker eval:
    correct = np.mean(y_test.argmax(axis=-1) == x_val_preds_adv)
    print('adversarial attack dropped validation accuracy to {}'.format(correct))

    # since DeepFool is not reproducible, saving the results in as numpy
    np.save(os.path.join(model_dir, 'X_val_adv.npy'), X_val_adv)
    np.save(os.path.join(model_dir, 'x_val_preds_adv.npy'), x_val_preds_adv)
    np.save(os.path.join(model_dir, 'x_val_features_adv.npy'), x_val_features_adv)
else:
    X_val_adv          = np.load(os.path.join(model_dir, 'X_val_adv.npy'))
    x_val_preds_adv    = np.load(os.path.join(model_dir, 'x_val_preds_adv.npy'))
    x_val_features_adv = np.load(os.path.join(model_dir, 'x_val_features_adv.npy'))

# what are the indices of the cifar10 set which the network succeeded classifying correctly,
# but the adversarial attack changed to a different class?
net_succ_attack_succ = []
net_succ_attack_succ_val_inds = []
for i, val_ind in enumerate(feeder.val_inds):
    net_succ    = x_val_preds[i] == y_val_sparse[i]
    attack_succ = x_val_preds[i] != x_val_preds_adv[i]
    if net_succ and attack_succ:
        net_succ_attack_succ.append(i)
        net_succ_attack_succ_val_inds.append(val_ind)
net_succ_attack_succ          = np.asarray(net_succ_attack_succ         , dtype=np.int32)
net_succ_attack_succ_val_inds = np.asarray(net_succ_attack_succ_val_inds, dtype=np.int32)

# verify everything is ok
if os.path.isfile(os.path.join(model_dir, 'net_succ_attack_succ.npy')):
    # assert match
    net_succ_attack_succ_old          = np.load(os.path.join(model_dir, 'net_succ_attack_succ.npy'))
    net_succ_attack_succ_val_inds_old = np.load(os.path.join(model_dir, 'net_succ_attack_succ_val_inds.npy'))
    assert (net_succ_attack_succ_old          == net_succ_attack_succ).all()
    assert (net_succ_attack_succ_val_inds_old == net_succ_attack_succ_val_inds).all()
else:
    np.save(os.path.join(model_dir, 'net_succ_attack_succ.npy')         , net_succ_attack_succ)
    np.save(os.path.join(model_dir, 'net_succ_attack_succ_val_inds.npy'), net_succ_attack_succ_val_inds)

# start the knn observation
knn = NearestNeighbors(n_neighbors=49000, p=2, n_jobs=20)
knn.fit(x_train_features)
all_neighbor_indices = knn.kneighbors(x_val_features, return_distance=False)

# setting up an adversarial feeder
adv_feeder = MyFeederValTest(rand_gen=rand_gen, as_one_hot=True, val_inds=feeder.val_inds, test_val_set=True)
adv_feeder.test_origin_data = X_val_adv
adv_feeder.test_data        = X_val_adv
adv_feeder.test_label       = one_hot(x_val_preds_adv, 10).astype(np.float32)

# now finding the influence
inspector_list     = []
inspector_adv_list = []
feeder.reset()
adv_feeder.reset()
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
train_iterations = 490  # was 500 wo validation

approx_params = {
    'scale': 200,
    'num_repeats': 5,
    'recursion_depth': 98,
    'recursion_batch_size': 100
}

def collect_influence(q, thread_id):
    while not q.empty():
        work = q.get()
        try:
            i = work[0]
            sub_val_index = net_succ_attack_succ[i]
            validation_index = feeder.val_inds[sub_val_index]
            assert validation_index == net_succ_attack_succ_val_inds[i]
            real_label = y_val_sparse[sub_val_index]
            adv_label  = x_val_preds_adv[sub_val_index]
            assert real_label != adv_label

            progress_str = '(thread_id={}) sample {}/{}: calculating scores for val index {} (sub={}). real label: {}, adv label: {}'\
                .format(thread_id, i+1, len(net_succ_attack_succ), validation_index, sub_val_index, _classes[real_label], _classes[adv_label])
            logging.info(progress_str)
            print(progress_str)

            for case in ['real', 'adv']:
                if case == 'real':
                    feed = feeder
                    insp = inspector_list[thread_id]
                elif case == 'adv':
                    feed = adv_feeder
                    insp = inspector_adv_list[thread_id]
                else:
                    raise AssertionError('only real and adv are accepted.')

                # creating the relevant index folders
                dir = os.path.join(model_dir, 'val_index_{}'.format(validation_index), case)
                if not os.path.exists(dir):
                    os.makedirs(dir)

                if FLAGS.prepare:
                    insp._prepare(
                        sess=sess,
                        test_indices=[sub_val_index],
                        test_batch_size=testset_batch_size,
                        approx_params=approx_params,
                        force_refresh=False
                    )
                    return

                scores = insp.upweighting_influence_batch(
                    sess=sess,
                    test_indices=[sub_val_index],
                    test_batch_size=testset_batch_size,
                    approx_params=approx_params,
                    train_batch_size=train_batch_size,
                    train_iterations=train_iterations)

                # save to disk
                np.save(os.path.join(dir, 'scores.npy'), scores)
                image = feed.val_inds[sub_val_index]
                np.save(os.path.join(dir, 'image.npy'), image)
        except Exception as e:
            print('Error with influence collect function for i={}: {}'.format(i, e))
            exit(1)
            raise AssertionError('Error with influence collect function for i={}!'.format(i))

        # signal to the queue that task has been processed
        q.task_done()
    return True

# set up a queue to hold all the jobs:
q = Queue(maxsize=0)
for i in range(len(net_succ_attack_succ)):
    q.put((i,))

for thread_id in range(FLAGS.num_threads):
    logging.info('Starting thread ', thread_id)
    worker = Thread(target=collect_influence, args=(q, thread_id))
    worker.setDaemon(True)
    worker.start()

q.join()
logging.info('All tasks completed.')
