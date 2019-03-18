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
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval

FLAGS = flags.FLAGS

NB_EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64


def cifar10_tutorial(train_start=0, train_end=60000, test_start=0,
                     test_end=10000, nb_epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                     learning_rate=LEARNING_RATE,
                     clean_train=CLEAN_TRAIN,
                     testing=False,
                     backprop_through_attack=BACKPROP_THROUGH_ATTACK,
                     nb_filters=NB_FILTERS, num_threads=None,
                     label_smoothing=0.1):
    """
    CIFAR10 cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                      before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                  to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                  example construction process during
                                  adversarial training.
    :param label_smoothing: float, amount of label smoothing for cross entropy
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

    # Train an MNIST model
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

    if clean_train:
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
              # optimizer=tf.train.MomentumOptimizer(learning_rate=train_params['learning_rate'],
              #                                      momentum=0.9,
              #                                      use_nesterov=True)
              optimizer=tf.train.GradientDescentOptimizer(learning_rate=train_params['learning_rate'])
              )

        save_path = os.path.join("model_save_dir", "model_checkpoint_lr_0.001_sgd.ckpt")
        saver = tf.train.Saver()
        saver.save(sess, save_path)

        # Calculate training error
        if testing:
            do_eval(preds, x_train, y_train, 'train_clean_train_clean_eval')

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_logits(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True)

        # Calculate training error
        if testing:
            do_eval(preds_adv, x_train, y_train, 'train_clean_train_adv_eval')

    return report


def main(argv=None):
    from cleverhans_tutorials import check_installation
    check_installation(__file__)

    cifar10_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', NB_FILTERS, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE, 'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                     ('If True, backprop through adversarial example '
                      'construction process during adversarial training'))

    tf.app.run()
