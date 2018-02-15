from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
from lib.base.agent_base import AgentBase

class Agent(AgentBase):
    __metaclass__ = ABCMeta

    def __init__(self, name, prm, model, dataset):
        super(Agent, self).__init__(name)
        self.prm     = prm
        self.model   = model
        self.dataset = dataset

        self.rand_gen = np.random.RandomState(self.prm.SUPERSEED)
        self.debug_mode            = self.prm.DEBUG_MODE

        self.eval_batch_size       = self.prm.train.train_control.EVAL_BATCH_SIZE
        self.root_dir              = self.prm.train.train_control.ROOT_DIR
        self.checkpoint_dir        = self.prm.train.train_control.CHECKPOINT_DIR

        # variables
        self.global_step = 0

    def build(self):
        """
        Building all trainer agents: train/validation/test sessions, file writers, retentions, hooks, etc.
        """
        self.model.build_graph()
        # self.print_model_info()
        self.dataset.build()

        self.saver = tf.train.Saver(max_to_keep=None, name=str(self), filename='model_ref')

        self.load_pretrained_from_ref()  # For loading params prior to setting monitored session

        self.build_retentions()

        self.build_train_env()
        self.build_validation_env()
        self.build_test_env()
        self.build_prediction_env()

        self.build_session()

        # creating train monitored session for automatically initializing the graph, running 'begin' function for
        # all hooks and using scaffold to finalize the graph. If there is a checkpoint already in the checkpoint_dir,
        # then it recovers the weights on the graph. Closing this session immediately after.
        self.finalize_graph()

        # Allow overwriting some parameters and optimizer on the graph from new parameter file.
        self.set_params()

        self.log.info('Done building agent {}'.format(str(self)))

    def build_retentions(self):
        # Retention for train/validation stats
        pass

    def build_train_env(self):
        pass

    def build_validation_env(self):
        pass

    def build_test_env(self):
        pass

    def build_prediction_env(self):
        pass

    @abstractmethod
    def build_session(self):
        """Create self.sess ans self.plain_sess"""
        pass

    def load_pretrained_from_ref(self):
        pass

    def finalize_graph(self):
        self.global_step = self.plain_sess.run(self.model.global_step)
        self.dataset.set_handles(self.plain_sess)

    def print_model_info(self):
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        self.total_parameters = param_stats.total_parameters
        self.log.info('total_params: {}\n'.format(self.total_parameters))

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    def set_params(self):
        """
        Overriding model's parameters if necessary.
        collecting the model values stored previously in the model.
        """

        [lrn_rate, xent_rate, weight_decay_rate, relu_leakiness, optimizer] = \
            self.plain_sess.run([self.model.lrn_rate, self.model.xent_rate, self.model.weight_decay_rate,
                           self.model.relu_leakiness, self.model.optimizer])

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

        self.plain_sess.run(assign_ops)

    def print_stats(self):
        '''print basic agent parameters'''
        super(Agent, self).print_stats()
        self.log.info(' EVAL_BATCH_SIZE: {}'.format(self.eval_batch_size))
        self.log.info(' ROOT_DIR: {}'.format(self.root_dir))
        self.log.info(' CHECKPOINT_DIR: {}'.format(self.checkpoint_dir))
        self.log.info(' DEBUG_MODE: {}'.format(self.debug_mode))



