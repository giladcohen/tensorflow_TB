from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np
import lib.logger.logger as logger
import tensorflow as tf

class TrainBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, prm, model, dataset):
        self.name = name
        self.log = logger.get_logger(name)
        self.prm     = prm
        self.model   = model
        self.dataset = dataset

        self.sess = None  # must be implemented in child

        # train parameters only
        self.trainer               = self.prm.train.train_control.TRAINER  # just used for printing.
        self.train_batch_size      = self.prm.train.train_control.TRAIN_BATCH_SIZE
        self.eval_batch_size       = self.prm.train.train_control.EVAL_BATCH_SIZE
        self.root_dir              = self.prm.train.train_control.ROOT_DIR
        self.train_dir             = self.prm.train.train_control.TRAIN_DIR
        self.eval_dir              = self.prm.train.train_control.EVAL_DIR
        self.checkpoint_dir        = self.prm.train.train_control.CHECKPOINT_DIR
        self.summary_steps         = self.prm.train.train_control.SUMMARY_STEPS
        self.checkpoint_secs       = self.prm.train.train_control.CHECKPOINT_SECS
        self.logger_steps          = self.prm.train.train_control.LOGGER_STEPS

    def __str__(self):
        return self.name

    def print_stats(self):
        '''print basic train parameters'''
        self.log.info('Train parameters:')
        self.log.info(' TRAINER: {}'.format(self.trainer))
        self.log.info(' TRAIN_BATCH_SIZE: {}'.format(self.train_batch_size))
        self.log.info(' EVAL_BATCH_SIZE: {}'.format(self.eval_batch_size))
        self.log.info(' ROOT_DIR: {}'.format(self.root_dir))
        self.log.info(' TRAIN_DIR: {}'.format(self.train_dir))
        self.log.info(' EVAL_DIR: {}'.format(self.eval_dir))
        self.log.info(' CHECKPOINT_DIR: {}'.format(self.checkpoint_dir))
        self.log.info(' SUMMARY_STEPS: {}'.format(self.summary_steps))
        self.log.info(' CHECKPOINT_SECS: {}'.format(self.checkpoint_secs))
        self.log.info(' LOGGER_STEPS: {}'.format(self.logger_steps))

    @abstractmethod
    def train(self):
        self.model.build_graph()

        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        self.log.info('total_params: {}\n'.format(param_stats.total_parameters))

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    def set_params(self):
        """Overriding model's parameters if necessary"""
        #model_variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
        #FIXME(gilad): automate this function for all names: model_variables[i].op.name

        # collecting the model values stored previously in the model:
        model_variables = [self.model.lrn_rate,
                           self.model.xent_rate,
                           self.model.weight_decay_rate,
                           self.model.relu_leakiness,
                           self.model.optimizer]
        dummy_feed_dict = self._get_dummy_feed() # tensorflow complains "no placeholder value" without this dummy feed
        [lrn_rate, xent_rate, weight_decay_rate, relu_leakiness, optimizer] = \
            self.sess.run(model_variables, feed_dict=dummy_feed_dict)
        assign_ops = []
        if not np.isclose(lrn_rate, self.prm.network.optimization.LEARNING_RATE):
            assign_ops.append(self.model.assign_ops['lrn_rate'])
            self.log.warning('changing model.lrn_rate from {} to {}'.
                             format(lrn_rate, self.prm.network.optimization.LEARNING_RATE))
        if not np.isclose(xent_rate, self.prm.network.optimization.XENTROPY_RATE):
            assign_ops.append(self.model.assign_ops['xent_rate'])
            self.log.warning('changing model.xent_rate from {} to {}'.
                             format(xent_rate, self.prm.network.optimization.XENTROPY_RATE))
        if not np.isclose(weight_decay_rate, self.prm.network.optimization.WEIGHT_DECAY_RATE):
            assign_ops.append(self.model.assign_ops['weight_decay_rate'])
            self.log.warning('changing model.weight_decay_rate from {} to {}'.
                             format(weight_decay_rate, self.prm.network.optimization.WEIGHT_DECAY_RATE))
        if not np.isclose(relu_leakiness, self.prm.network.system.RELU_LEAKINESS):
            assign_ops.append(self.model.assign_ops['relu_leakiness'])
            self.log.warning('changing model.relu_leakiness from {} to {}'.
                             format(relu_leakiness, self.prm.network.system.RELU_LEAKINESS))
        if optimizer != self.prm.network.optimization.OPTIMIZER:
            assign_ops.append(self.model.assign_ops['optimizer'])
            self.log.warning('changing model.optimizer from {} to {}'.
                             format(optimizer, self.prm.network.optimization.OPTIMIZER))
        self.sess.run(assign_ops)

    def _get_dummy_feed(self):
        """Getting dummy feed to bypass tensorflow (possible bug?) complaining about no placeholder value"""
        images, labels = self.dataset.get_mini_batch_train(batch_size=1)
        feed_dict = {self.model.images     : images,
                     self.model.labels     : labels,
                     self.model.is_training: False}
        return feed_dict
