from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
import os

from cleverhans.attacks import FastGradientMethod
from cleverhans.augmentation import random_horizontal_flip, random_shift
from tensorflow.python.platform import flags
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
from cleverhans.loss import CrossEntropy, WeightDecay, WeightedSum
from tensorflow_TB.lib.models.darkon_replica_model import DarkonReplica
from tensorflow_TB.cleverhans_alias.train_alias import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval
from tensorflow_TB.lib.datasets.influence_feeder_val import MyFeederVal

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 200, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 125, 'Size of training batches')
flags.DEFINE_float('weight_decay', 0.0004, 'weight decay')
flags.DEFINE_string('optimizer', 'mom', 'optimizer')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('lr_factor', 0.9, 'A factor to decay a learning rate')
flags.DEFINE_integer('lr_patience', 3, 'epochs with no metric improvements')
flags.DEFINE_integer('lr_cooldown', 2, 'epochs in refractory period')
flags.DEFINE_string('checkpoint_name', '', 'checkpoint name')
flags.DEFINE_float('label_smoothing', 0.1, 'label smoothing')


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

feeder = MyFeederVal(as_one_hot=True, rand_gen=rand_gen)

# get the data
X_train, y_train = feeder.train_indices(range(49000))
X_val, y_val     = feeder.test_indices(range(1000))
y_train_sparse   = y_train.argmax(axis=-1).astype(np.int32)
y_val_sparse     = y_val.argmax(axis=-1).astype(np.int32)

dataset_size  = X_train.shape[0]
dataset_train = tf.data.Dataset.range(dataset_size)
dataset_train = dataset_train.shuffle(4096)
dataset_train = dataset_train.repeat()

def lookup(p):
    return X_train[p], y_train[p]
dataset_train = dataset_train.map(lambda i: tf.py_func(lookup, [i], [tf.float32] * 2))

dataset_train = dataset_train.map(lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
dataset_train = dataset_train.batch(FLAGS.batch_size)
dataset_train = dataset_train.prefetch(16)

# Use Image Parameters
img_rows, img_cols, nchannels = X_val.shape[1:4]
nb_classes = y_val.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

model_dir = os.path.join('/data/gilad/logs/influence', FLAGS.checkpoint_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Train an CIFAR-10 model
train_params = {
    'nb_epochs': FLAGS.nb_epochs,
    'batch_size': FLAGS.batch_size,
    'learning_rate': FLAGS.learning_rate,
    'lr_factor': FLAGS.lr_factor,
    'lr_patience': FLAGS.lr_patience,
    'lr_cooldown': FLAGS.lr_cooldown,
    'best_model_path': os.path.join(model_dir, 'best_model.ckpt')
}
eval_params = {'batch_size': FLAGS.batch_size}
fgsm_params = {
    'eps': 0.3,
    'clip_min': 0.,
    'clip_max': 1.
}

model = DarkonReplica(scope='model1', nb_classes=10, n=5, input_shape=[32, 32, 3])
logits = model.get_logits(x)
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

def evaluate():
    return do_eval(logits, X_val, y_val, 'clean_train_clean_eval', False)


train(sess, full_loss, None, None,
      dataset_train=dataset_train, dataset_size=dataset_size,
      evaluate=evaluate, args=train_params, rng=rand_gen,
      var_list=model.get_params(),
      optimizer=FLAGS.optimizer)

save_path = os.path.join(model_dir, "model_checkpoint.ckpt")
saver = tf.train.Saver()
saver.save(sess, save_path, global_step=tf.train.get_global_step())
np.save(os.path.join(model_dir, "val_indices.npy"))

# Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
fgsm = FastGradientMethod(model, sess=sess)
adv_x = fgsm.generate(x, **fgsm_params)
logits_adv = model.get_logits(adv_x)

# Evaluate the accuracy of the CIFAR-10 model on adversarial examples
do_eval(logits_adv, X_val, y_val, 'clean_train_adv_eval', True)

print('done')