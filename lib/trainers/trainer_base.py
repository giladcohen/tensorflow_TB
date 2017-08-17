from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
from lib.base.agent_base import AgentBase
from lib.base.collections import TRAIN_SUMMARIES
import utils
from lib.retention import Retention
from utils.tensorboard_logging import TBLogger


class TrainerBase(AgentBase):
    __metaclass__ = ABCMeta

    def __init__(self, name, prm, model, dataset):
        super(TrainerBase, self).__init__(name)
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
        self.eval_steps = self.prm.train.train_control.EVAL_STEPS
        self.evals_in_epoch = self.prm.train.train_control.EVALS_IN_EPOCH
        self.skip_first_evaluation = self.prm.train.train_control.SKIP_FIRST_EVALUATION
        if self.eval_steps is None:
            self.log.warning('EVAL_STEPS is None. Setting EVAL_STEPS based on EVALS_IN_EPOCH (for initial pool size)')
            self.eval_steps = int(self.dataset.train_dataset.pool_size() / (self.train_batch_size * self.evals_in_epoch))

        # variables
        self.global_step    = 0
        if self.skip_first_evaluation:
            self.log.info('skipping evaluation for global_step=0')
            self._activate_eval = False
        else:
            self._activate_eval = True

        # agents
        self.sess = None
        self.Factories = utils.factories.Factories(self.prm)  # to get hooks
        self.retention = Retention('Retention', self.prm)  # for logging and setting lrn rate
        self.learning_rate_hook = self.Factories.get_learning_rate_setter(self.model, self.dataset.train_dataset, self.retention)

        self.model.build_graph()
        self.print_model_info()

        # setting file writers
        self.summary_writer_train = tf.summary.FileWriter(self.train_dir)  # for training
        self.summary_writer_eval  = tf.summary.FileWriter(self.eval_dir)   # for evaluation
        self.tb_logger_eval = TBLogger(self.summary_writer_eval)

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
        self.log.info(' EVAL_STEPS: {}'.format(self.eval_steps))
        self.log.info(' EVALS_IN_EPOCH: {}'.format(self.evals_in_epoch))
        self.log.info(' SKIP_FIRST_EVALUATION: {}'.format(self.skip_first_evaluation))
        self.retention.print_stats()
        self.learning_rate_hook.print_stats()

    def get_train_summaries(self):
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('score', self.score))

    def calculate_score(self):
        """Calculating the score of the network. It is model.score by default"""
        return self.model.score

    def train(self):
        self.score = self.calculate_score()

        self.get_train_summaries()

        summary_hook = tf.train.SummarySaverHook(
            save_steps=self.summary_steps,
            summary_writer=self.summary_writer_train,
            summary_op=tf.summary.merge([self.model.summaries] + tf.get_collection(TRAIN_SUMMARIES)))

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': self.model.global_step,
                     'loss_xent': self.model.xent_cost,
                     'loss_wd': self.model.wd_cost,
                     'loss': self.model.cost,
                     'score': self.score},
            every_n_iter=self.logger_steps)

        self.sess = tf.train.MonitoredTrainingSession(
            checkpoint_dir=self.checkpoint_dir,
            hooks=[logging_hook, self.learning_rate_hook],
            chief_only_hooks=[summary_hook],
            save_checkpoint_secs=self.checkpoint_secs,
            config=tf.ConfigProto(allow_soft_placement=True))

        self.set_params()

        while not self.sess.should_stop():
            if self.global_step % self.eval_steps == 0 and self._activate_eval:
                self.eval_step()
                self._activate_eval = False
            else:
                self.train_step()
                self._activate_eval = True

    def print_model_info(self):
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
        images, labels = self.dataset.get_mini_batch_train(indices=[0])
        feed_dict = {self.model.images     : images,
                     self.model.labels     : labels,
                     self.model.is_training: False}
        return feed_dict

    @abstractmethod
    def train_step(self):
        '''Implementing one training step. Must update self.global_step.'''
        pass

    @abstractmethod
    def eval_step(self):
        '''Implementing one evaluation step.'''
        pass
