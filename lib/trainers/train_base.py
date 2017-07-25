from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
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
        self.evals_in_epoch        = self.prm.train.train_control.EVALS_IN_EPOCH

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
        self.log.info(' EVALS_IN_EPOCH: {}'.format(self.evals_in_epoch))

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
