from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from lib.base.agent import Agent
from lib.base.collections import TRAIN_SUMMARIES
import utils
from lib.retention import Retention
from lib.trainers.hooks.global_step_checkpoint_saver_hook import GlobalStepCheckpointSaverHook
from lib.trainers.hooks.train_summary_saver_hook import TrainSummarySaverHook
from utils.tensorboard_logging import TBLogger

class TrainerBase(Agent):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(TrainerBase, self).__init__(*args, **kwargs)

        # train parameters only
        self.trainer               = self.prm.train.train_control.TRAINER  # just used for printing.
        self.train_dir             = self.prm.train.train_control.TRAIN_DIR
        self.eval_dir              = self.prm.train.train_control.EVAL_DIR
        self.pred_dir              = self.prm.train.train_control.PREDICTION_DIR
        self.test_dir              = self.prm.train.train_control.TEST_DIR
        self.summary_steps         = self.prm.train.train_control.SUMMARY_STEPS
        self.checkpoint_secs       = self.prm.train.train_control.CHECKPOINT_SECS
        self.checkpoint_steps      = self.prm.train.train_control.CHECKPOINT_STEPS
        self.last_step             = self.prm.train.train_control.LAST_STEP
        self.logger_steps          = self.prm.train.train_control.LOGGER_STEPS
        self.eval_steps            = self.prm.train.train_control.EVAL_STEPS
        self.test_steps            = self.prm.train.train_control.TEST_STEPS

        self.skip_first_evaluation = self.prm.train.train_control.SKIP_FIRST_EVALUATION
        if self.last_step is None:
            self.log.warning('LAST_STEP is None. Setting LAST_STEP=1000000')
            self.last_step = 1000000
        self.Factories = utils.factories.Factories(self.prm)  # to get hooks

        # variables
        if self.skip_first_evaluation:
            self.log.info('skipping evaluation for global_step=0')
            self._activate_eval = False
            self._activate_test = False
        else:
            self._activate_eval = True
            self._activate_test = True

    def load_pretrained_from_ref(self):
        pass

    def build_retentions(self):
        # Retention for train/validation stats
        self.train_retention      = Retention('retention_train'     , self.prm)
        self.validation_retention = Retention('retention_validation', self.prm)
        self.test_retention       = Retention('retention_test'      , self.prm)

    def build_train_env(self):
        self.log.info("Starting building the train environment")
        self.summary_writer_train = tf.summary.FileWriter(self.train_dir)
        self.tb_logger_train = TBLogger(self.summary_writer_train)
        self.get_train_summaries()

        self.learning_rate_hook = self.Factories.get_learning_rate_setter(self.model, self.validation_retention)

        summary_hook = TrainSummarySaverHook(
            name='train_summary_saver_hook',
            prm=self.prm,
            model=self.model,
            save_steps=self.summary_steps,
            summary_writer=self.summary_writer_train,
            summary_op=tf.summary.merge([self.model.summaries] + tf.get_collection(TRAIN_SUMMARIES)))

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': self.model.global_step,
                     'loss-loss_wd': self.model.cost - self.model.wd_cost,
                     'loss_wd': self.model.wd_cost,
                     'loss': self.model.cost,
                     'score': self.model.score},
            every_n_iter=self.logger_steps)

        checkpoint_hook = GlobalStepCheckpointSaverHook(
            name='global_step_checkpoint_saver_hook',
            prm=self.prm,
            model=self.model,
            steps_to_save=self.checkpoint_steps,
            checkpoint_dir=self.checkpoint_dir,
            saver=self.saver,
            checkpoint_basename='model_schedule.ckpt')

        auto_checkpoint_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=self.checkpoint_dir,
            save_secs=self.checkpoint_secs,
            saver=tf.train.Saver(max_to_keep=1, name='auto_saver'))

        stop_at_step_hook = tf.train.StopAtStepHook(last_step=self.last_step)

        self.train_session_hooks = [summary_hook   , logging_hook        , self.learning_rate_hook,
                                    checkpoint_hook, auto_checkpoint_hook, stop_at_step_hook]

    def build_validation_env(self):
        self.log.info("Starting building the validation environment")
        self.summary_writer_validation = tf.summary.FileWriter(self.eval_dir)
        self.tb_logger_validation = TBLogger(self.summary_writer_validation)

    def build_test_env(self):
        self.log.info("Starting building the test environment")
        self.summary_writer_test = tf.summary.FileWriter(self.test_dir)
        self.tb_logger_test = TBLogger(self.summary_writer_test)

    def build_prediction_env(self):
        self.log.info("Starting building the prediction environment")
        self.summary_writer_pred = tf.summary.FileWriter(self.pred_dir)
        self.tb_logger_pred = TBLogger(self.summary_writer_pred)

    def build_session(self):
        # create session
        self.sess = tf.train.MonitoredTrainingSession(
            checkpoint_dir=self.checkpoint_dir,
            hooks=self.train_session_hooks,
            save_checkpoint_secs=self.checkpoint_secs,
            config=tf.ConfigProto(allow_soft_placement=True))

        self.plain_sess = self.sess._tf_sess()

    def print_stats(self):
        self.log.info('Train parameters:')
        super(TrainerBase, self).print_stats()
        self.log.info(' TRAINER: {}'.format(self.trainer))
        self.log.info(' TRAIN_DIR: {}'.format(self.train_dir))
        self.log.info(' EVAL_DIR: {}'.format(self.eval_dir))
        self.log.info(' TEST_DIR: {}'.format(self.test_dir))
        self.log.info(' PREDICTION_DIR: {}'.format(self.pred_dir))
        self.log.info(' SUMMARY_STEPS: {}'.format(self.summary_steps))
        self.log.info(' CHECKPOINT_SECS: {}'.format(self.checkpoint_secs))
        self.log.info(' CHECKPOINT_STEPS: {}'.format(self.checkpoint_steps))
        self.log.info(' LAST_STEP: {}'.format(self.last_step))
        self.log.info(' LOGGER_STEPS: {}'.format(self.logger_steps))
        self.log.info(' EVAL_STEPS: {}'.format(self.eval_steps))
        self.log.info(' TEST_STEPS: {}'.format(self.test_steps))
        self.log.info(' SKIP_FIRST_EVALUATION: {}'.format(self.skip_first_evaluation))
        self.log.info(' DEBUG_MODE: {}'.format(self.debug_mode))
        self.train_retention.print_stats()
        self.validation_retention.print_stats()
        self.test_retention.print_stats()
        self.learning_rate_hook.print_stats()

    def get_train_summaries(self):
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('score', self.model.score))
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('weight_decay_rate', self.model.weight_decay_rate))
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('dropout_keep_prob', self.model.dropout_keep_prob))

    def train(self):
        while not self.sess.should_stop():
            if self.to_eval():
                self.eval_step()
                self._activate_eval = False
            elif self.to_test():
                self.test_step()
                self._activate_test = False
            else:
                self.train_step()
                self._activate_eval = True
                self._activate_test = True
        self.log.info('Stop training at global_step={}'.format(self.global_step))

    def print_model_info(self):
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        self.total_parameters = param_stats.total_parameters
        self.log.info('total_params: {}\n'.format(self.total_parameters))

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    @abstractmethod
    def train_step(self):
        '''Implementing one training step. Must update self.global_step.'''
        pass

    @abstractmethod
    def eval_step(self):
        '''Implementing one evaluation step.'''
        pass

    @abstractmethod
    def test_step(self):
        """Implementing one test step."""
        pass

    def to_eval(self):
        return self.global_step % self.eval_steps == 0 and self._activate_eval and self.dataset.validation_set_size > 0

    def to_test(self):
        return self.global_step % self.test_steps == 0 and self._activate_test
