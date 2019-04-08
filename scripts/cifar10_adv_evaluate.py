from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import os

import darkon
from cleverhans.attacks import FastGradientMethod
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

FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_float('weight_decay', 0.0004, 'weight decay')
flags.DEFINE_string('checkpoint_name', '', 'checkpoint name')
flags.DEFINE_float('label_smoothing', 0.1, 'label smoothing')
flags.DEFINE_string('workspace', 'influence_workspace_310319', 'workspace dir')

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
val_indices   = np.load(os.path.join(model_dir, 'val_indices.npy'))  # get the original validation indices

feeder = MyFeederValTest(as_one_hot=True, val_inds=val_indices, test_val_set=True)

# get the data
val_indices = np.load(os.path.join(model_dir, 'val_indices.npy'))  # get the original validation indices
X_train, y_train = feeder.train_indices(range(50000))

X_test, y_test   = feeder.test_indices(range(10000))
y_train_sparse   = y_train.argmax(axis=-1).astype(np.int32)
y_test_sparse    = y_test.argmax(axis=-1).astype(np.int32)

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
save_path = os.path.join("model_save_dir", "model_checkpoint_{}.ckpt-80000".format(FLAGS.checkpoint_name))
saver = tf.train.Saver()
saver.restore(sess, save_path)

# predict labels from trainset
x_train_preds, x_train_features = np_evaluate(sess, [preds, embeddings], X_train, y_train, x, y, FLAGS.batch_size, log=logging)
x_train_preds = x_train_preds.astype(np.int32)
do_eval(logits, X_train, y_train, 'clean_train_clean_eval_trainset', False)
# predict labels from testset
x_test_preds, x_test_features = np_evaluate(sess, [preds, embeddings], X_test, y_test, x, y, FLAGS.batch_size, log=logging)
x_test_preds = x_test_preds.astype(np.int32)
do_eval(logits, X_test, y_test, 'clean_train_clean_eval_testset', False)
#np.mean(y_test.argmax(axis=-1) == x_test_preds)

# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
fgsm = FastGradientMethod(model, sess=sess)
adv_x = fgsm.generate(x, **fgsm_params)
preds_adv      = model.get_predicted_class(adv_x)
logits_adv     = model.get_logits(adv_x)
embeddings_adv = model.get_embeddings(adv_x)

# Evaluate the accuracy of the CIFAR-10 model on adversarial examples
X_test_adv, x_test_preds_adv, x_test_features_adv = np_evaluate(sess, [adv_x, preds_adv, embeddings_adv], X_test, y_test, x, y, FLAGS.batch_size, log=logging)
x_test_preds_adv = x_test_preds_adv.astype(np.int32)
do_eval(logits_adv, X_test, y_test, 'clean_train_adv_eval', True)

# what are the indices of the test set which the network succeeded classifying correctly,
# but the FGSM attack changed to a different class?
net_succ_attack_succ = []
for i in range(len(X_test)):
    net_succ    = x_test_preds[i] == y_test_sparse[i]
    attack_succ = x_test_preds[i] != x_test_preds_adv[i]
    if net_succ and attack_succ:
        net_succ_attack_succ.append(i)

# Due to lack of time, we can also sample 5 inputs of each class. Here we randomly select them...
test_indices = []
for cls in range(len(_classes)):
    cls_test_indices = []
    got_so_far = 0
    while got_so_far < 5:
        cls_test_index = rand_gen.choice(np.where(y_test_sparse == cls)[0])
        if cls_test_index in net_succ_attack_succ:
            cls_test_indices.append(cls_test_index)
            got_so_far += 1
    test_indices.extend(cls_test_indices)

# optional: divide test indices
# test_indices = test_indices[b:e]

# start the knn observation
knn = NearestNeighbors(n_neighbors=50000, p=2, n_jobs=20)
knn.fit(x_train_features)
all_neighbor_indices = knn.kneighbors(x_test_features, return_distance=False)

# now finding the influence
feeder.reset()

inspector = darkon.Influence(
    workspace=FLAGS.workspace,
    feeder=feeder,
    loss_op_train=full_loss.fprop(x=x, y=y),
    loss_op_test=loss.fprop(x=x, y=y),
    x_placeholder=x,
    y_placeholder=y)

# setting up an adversarial feeder
adv_feeder = MyFeeder(as_one_hot=True)
adv_feeder.test_origin_data = X_test_adv
adv_feeder.test_data = X_test_adv
adv_feeder.test_label = one_hot(x_test_preds_adv, 10).astype(np.float32)
adv_feeder.reset()

inspector_adv = darkon.Influence(
    workspace=FLAGS.workspace,
    feeder=adv_feeder,
    loss_op_train=full_loss.fprop(x=x, y=y),
    loss_op_test=loss.fprop(x=x, y=y),
    x_placeholder=x,
    y_placeholder=y)

testset_batch_size = 100
train_batch_size = 100
train_iterations = 500

approx_params = {
    'scale': 200,
    'num_repeats': 5,
    'recursion_depth': 100,
    'recursion_batch_size': 100
}

for i, test_index in enumerate(test_indices):
    real_label = y_test_sparse[test_index]
    adv_label  = x_test_preds_adv[test_index]

    logging.info("sample {}/{}: calculating scores for test index {}. real label: {}, adv label: {}"
                 .format(i+1, len(test_indices), test_index, _classes[real_label], _classes[adv_label]))

    for case in ['real', 'adv']:
        if case == 'real':
            insp = inspector
        elif case == 'adv':
            insp = inspector_adv
        else:
            raise AssertionError('only real and adv are accepted.')

        # creating the relevant index folders
        dir = os.path.join(FLAGS.workspace, 'test_index_{}'.format(test_index), case)
        if not os.path.exists(dir):
            os.makedirs(dir)

        scores = insp.upweighting_influence_batch(
            sess=sess,
            test_indices=[test_index],
            test_batch_size=testset_batch_size,
            approx_params=approx_params,
            train_batch_size=train_batch_size,
            train_iterations=train_iterations,
            force_refresh=True)

        sorted_indices = np.argsort(scores)
        harmful = sorted_indices[:50]
        helpful = sorted_indices[-50:][::-1]

        # have some figures
        cnt_harmful_in_knn = 0
        print('\nHarmful:')
        for idx in harmful:
            print('[{}] {}'.format(idx, scores[idx]))
            if idx in all_neighbor_indices[test_index, 0:50]:
                cnt_harmful_in_knn += 1
        harmful_summary_str = '{}: {} out of {} harmful images are in the {}-NN\n'.format(case, cnt_harmful_in_knn, len(harmful), 50)
        print(harmful_summary_str)

        cnt_helpful_in_knn = 0
        print('\nHelpful:')
        for idx in helpful:
            print('[{}] {}'.format(idx, scores[idx]))
            if idx in all_neighbor_indices[test_index, 0:50]:
                cnt_helpful_in_knn += 1
        helpful_summary_str = '{}: {} out of {} helpful images are in the {}-NN\n'.format(case, cnt_helpful_in_knn, len(helpful), 50)
        print(helpful_summary_str)

        fig, axes1 = plt.subplots(5, 10, figsize=(30, 10))
        target_idx = 0
        for j in range(5):
            for k in range(10):
                idx = all_neighbor_indices[test_index, target_idx]
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X_train[idx])
                label_str = _classes[y_train_sparse[idx]]
                axes1[j][k].set_title('[{}]: {}'.format(idx, label_str))
                target_idx += 1
        plt.savefig(os.path.join(FLAGS.workspace, 'test_index_{}'.format(test_index), case, 'nearest_neighbors.png'), dpi=350)
        plt.close()

        total_rank = 0.0
        fig, axes1 = plt.subplots(5, 10, figsize=(30, 10))
        target_idx = 0
        for j in range(5):
            for k in range(10):
                idx = helpful[target_idx]
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X_train[idx])
                label_str = _classes[y_train_sparse[idx]]
                loc_in_knn = np.where(all_neighbor_indices[test_index] == idx)[0][0]
                total_rank += loc_in_knn
                axes1[j][k].set_title('[{}]: {} #nn:{}'.format(idx, label_str, loc_in_knn))
                target_idx += 1
        label_rank = total_rank / 50
        plt.savefig(os.path.join(FLAGS.workspace, 'test_index_{}'.format(test_index), case, 'helpful.png'), dpi=350)
        plt.close()

        fig, axes1 = plt.subplots(5, 10, figsize=(30, 10))
        target_idx = 0
        for j in range(5):
            for k in range(10):
                idx = harmful[target_idx]
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(X_train[idx])
                label_str = _classes[y_train_sparse[idx]]
                loc_in_knn = np.where(all_neighbor_indices[test_index] == idx)[0][0]
                axes1[j][k].set_title('[{}]: {} #nn:{}'.format(idx, label_str, loc_in_knn))
                target_idx += 1
        plt.savefig(os.path.join(FLAGS.workspace, 'test_index_{}'.format(test_index), case, 'harmful.png'), dpi=350)
        plt.close()

        # save to disk
        np.save(os.path.join(FLAGS.workspace, 'test_index_{}'.format(test_index), case, 'scores'), scores)
        if case == 'real':
            image = X_test[test_index]
        else:
            image = X_test_adv[test_index]
        np.save(os.path.join(FLAGS.workspace, 'test_index_{}'.format(test_index), case, 'image'), image)

        # getting two ranks - one rank for the real label and another rank for the adv label.
        # what is a "rank"?
        # A rank is the average nearest neighbor location of all the helpful training indices.
        with open(os.path.join(FLAGS.workspace, 'test_index_{}'.format(test_index), case, 'summary.txt'), 'w+') as f:
            f.write(harmful_summary_str)
            f.write(helpful_summary_str)
            f.write('label ({} -> {}) {} rank: {}'.format(_classes[real_label], _classes[adv_label], case, label_rank))
