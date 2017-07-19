from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import lib.logger.logger as logger
import os
import tensorflow as tf
import sys

class TrainBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, prm, model, dataset, logger_name):
        self._logger = logger.get_logger(logger_name)

        with tf.Graph().as_default():
            self.prm     = prm
            self.model   = model
            self.dataset = dataset

            # train parameters only
            self.learning_rate      = self.prm.train.train_control.LEARNING_RATE
            self.train_batch_size   = self.prm.train.train_control.TRAIN_BATCH_SIZE
            self.eval_batch_size    = self.prm.train.train_control.EVAL_BATCH_SIZE
            self.num_gpus           = self.prm.train.train_control.NUM_GPUS
            self.root_dir           = self.prm.train.train_control.ROOT_DIR
            self.train_dir          = self.prm.train.train_control.TRAIN_DIR
            self.eval_dir           = self.prm.train.train_control.EVAL_DIR
            self.checkpoint_dir     = self.prm.train.train_control.CHECKPOINT_DIR
            self.max_steps          = self.prm.train.train_control.MAX_STEPS
            self.summary_steps      = self.prm.train.train_control.SUMMARY_STEPS
            self.checkpoint_secs    = self.prm.train.train_control.CHECKPOINT_SECS
            self.logger_steps       = self.prm.train.train_control.LOGGER_STEPS
            self.eval_steps         = self.prm.train.train_control.EVAL_STEPS

            self.print_stats()

    def print_stats(self):
        '''print basic train parameters'''
        self._logger.info('Train parameters:')
        self._logger.info(' LEARNING_RATE: {}'.format(self.learning_rate))
        self._logger.info(' TRAIN_BATCH_SIZE: {}'.format(self.train_batch_size))
        self._logger.info(' EVAL_BATCH_SIZE: {}'.format(self.eval_batch_size))
        self._logger.info(' NUM_GPUS: {}'.format(self.num_gpus))
        self._logger.info(' ROOT_DIR: {}'.format(self.root_dir))
        self._logger.info(' TRAIN_DIR: {}'.format(self.train_dir))
        self._logger.info(' EVAL_DIR: {}'.format(self.eval_dir))
        self._logger.info(' CHECKPOINT_DIR: {}'.format(self.checkpoint_dir))
        self._logger.info(' MAX_STEPS: {}'.format(self.max_steps))
        self._logger.info(' SUMMARY_STEPS: {}'.format(self.summary_steps))
        self._logger.info(' CHECKPOINT_SECS: {}'.format(self.checkpoint_secs))
        self._logger.info(' LOGGER_STEPS: {}'.format(self.logger_steps))
        self._logger.info(' EVAL_STEPS: {}'.format(self.eval_steps))

    @abstractmethod
    def train(self):
        self.model.build_graph()

        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        sys.stdout.write('total_params: %d\n' % param_stats.total_parameters) #FIXME(print with logger DEBUG)

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS) #FIXME(print with logger DEBUG)








