"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
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
# from cleverhans.compat import flags
from tensorflow.python.platform import flags
from cleverhans.dataset import CIFAR10
from cleverhans.loss import CrossEntropy, WeightDecay, WeightedSum
# from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from tensorflow_TB.lib.models.darkon_replica_model import DarkonReplica
# from cleverhans.train import train
from tensorflow_TB.cleverhans_alias.train_alias import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval

FLAGS = flags.FLAGS

NB_EPOCHS = 200
BATCH_SIZE = 128
OPTIMIZER = 'sgd'
LEARNING_RATE = 0.001
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64
CHECKPOINT_NAME = ''


def cifar10_tutorial(train_start=0, train_end=60000, test_start=0,
                     test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                     optimizer=OPTIMIZER, learning_rate=LEARNING_RATE,
                     num_threads=None, checkpoint_name=CHECKPOINT_NAME,
                     label_smoothing=0.1):
    """
    CIFAR10 cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param optimizer: tf.train.optimizer
    :param learning_rate: learning rate for training
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :param num_threads: num of threads
    :param checkpoint_name: checkpoint suffix
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1, allow_soft_placement=True)
    else:
        config_args = dict(allow_soft_placement=True)
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get CIFAR10 data
    data = CIFAR10(train_start=train_start, train_end=train_end,
                   test_start=test_start, test_end=test_end)
    dataset_size = data.x_train.shape[0]
    dataset_train = data.to_tensorflow()[0]
    dataset_train = dataset_train.map(lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(16)
    x_train, y_train = data.get_set('train')
    x_test, y_test = data.get_set('test')

    # Use Image Parameters
    img_rows, img_cols, nchannels = x_test.shape[1:4]
    nb_classes = y_test.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Train an CIFAR-10 model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    eval_params = {'batch_size': batch_size}
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    rng = np.random.RandomState([2017, 8, 30])

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

    # model = make_wresnet(nb_classes=nb_classes, input_shape=(None, 32, 32, 3))  # TODO(add scope)
    model = DarkonReplica(scope='model1', nb_classes=10, n=5, input_shape=[32, 32, 3])

    preds = model.get_logits(x)
    loss = CrossEntropy(model, smoothing=label_smoothing)
    regu_losses = WeightDecay(model)
    full_loss = WeightedSum(model, [(1.0, loss), (0.0002, regu_losses)])

    def evaluate():
        do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)

    train(sess, full_loss, None, None,
          dataset_train=dataset_train, dataset_size=dataset_size,
          evaluate=evaluate, args=train_params, rng=rng,
          var_list=model.get_params(),
          optimizer=optimizer)

    save_path = os.path.join("model_save_dir",
                             "model_checkpoint_{}.ckpt".format(checkpoint_name))
    saver = tf.train.Saver()
    saver.save(sess, save_path, global_step=tf.train.get_global_step())

    # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
    fgsm = FastGradientMethod(model, sess=sess)
    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model.get_logits(adv_x)

    # Evaluate the accuracy of the CIFAR-10 model on adversarial examples
    do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True)

    return report


def main(argv=None):
    from cleverhans_tutorials import check_installation
    check_installation(__file__)

    cifar10_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                     optimizer=FLAGS.optimizer, learning_rate=FLAGS.learning_rate,
                     checkpoint_name=FLAGS.checkpoint_name)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate for training')
    flags.DEFINE_string('optimizer', OPTIMIZER, 'optimizer')
    flags.DEFINE_string('checkpoint_name', CHECKPOINT_NAME, 'optimizer')

    tf.app.run()
