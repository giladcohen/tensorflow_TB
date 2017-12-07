from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
import numpy as np
import tensorflow as tf
from lib.base.collections import TRAIN_SUMMARIES
from lib.retention import Retention
from lib.trainers.hooks.global_step_checkpoint_saver_hook import GlobalStepCheckpointSaverHook
from utils.tensorboard_logging import TBLogger


class DynamicModelTrainer(ActiveTrainerBase):

    def train(self):
        while True:
            if self.to_annotate():
                self.annot_step()
                self.update_model()
                self._activate_annot = False
            elif self.to_eval():
                self.eval_step()
                self._activate_eval  = False
            else:
                self.train_step()
                self._activate_annot = True
                self._activate_eval  = True

    def update_model(self):
        lp = self.dataset.train_dataset.pool_size()
        global_step_copy = self.global_step
        # [16, 160, 320, 640] for 50k samples. for 1k: [2, 4, 6, 14]. weight_decay for 1k: 0.0390625
        if lp == 2000:
            resnet_filters    = np.array([2, 6, 14, 26])
            weight_decay_rate = 0.01953125
            self.pca_embedding_dims = 5
        elif lp == 3000:
            resnet_filters = np.array([2, 10, 20, 38])
            weight_decay_rate = 0.01302083333
            self.pca_embedding_dims = 8
        elif lp == 4000:
            resnet_filters = np.array([2, 14, 26, 52])
            weight_decay_rate = 0.009765625
            self.pca_embedding_dims = 10
        elif lp == 5000:
            resnet_filters = np.array([2, 16, 32, 64])
            weight_decay_rate = 0.0078125
            self.pca_embedding_dims = 13
        else:
            err_str = 'pool size is {}. This is not possible'.format(lp)
            self.log.error(err_str)
            raise AssertionError(err_str)
        self.log.info('getting a new model for lp={}. resnet_filters={}. weight_decay_rate={}. pca_embedding_dims={}'
                      .format(lp, resnet_filters, weight_decay_rate, self.pca_embedding_dims))

        tf.reset_default_graph()
        update_ops_collection      = tf.get_collection_ref(tf.GraphKeys.UPDATE_OPS)
        assertions_collection      = tf.get_collection_ref('assertions')
        losses_collection          = tf.get_collection_ref('losses')
        train_summaries_collection = tf.get_collection_ref(TRAIN_SUMMARIES)

        # debug
        self.log.info(' update_ops_collection = {}\n assertions_collection = {}\n'
                      ' losses_collection = {}\n train_summaries_collection = {}'
                      .format(update_ops_collection, assertions_collection, losses_collection, train_summaries_collection))

        del update_ops_collection[:]
        del assertions_collection[:]
        del losses_collection[:]

        # overwrite the global step
        self.log.info('overwriting graph\'s global step to {}'.format(self.global_step))
        global_step_tensor = tf.train.get_or_create_global_step()
        update_global_step = global_step_tensor.assign(self.global_step)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(update_global_step)

        # debug
        global_step_value = sess.run(global_step_tensor)
        self.log.info('DEBUG: global_step_value = {}'.format(global_step_value))

        self.model = self.Factories.get_model()
        self.model.resnet_filters = resnet_filters

        self.embedding_dims = resnet_filters[-1]
        self._activate_eval = False  # cannot evaluate fresh model without assigning model variables. train first.
        self.mode = None
        self.sess = None
        self.model.build_graph()
        self.print_model_info()

        self.learning_rate_hook = self.Factories.get_learning_rate_setter(self.model, self.dataset.train_dataset, self.validation_retention)
        self.build_train_env()
        self.finalize_graph()  # to create the new monitored session and feeding initial dummy dict

        # just to re-initialize the graph
        # self.log.info('resetting the global_step to={}'.format(self.global_step))
        # self.sess.run(self.model.assign_ops['global_step_ow'], feed_dict={self.model.global_step_ph: self.global_step})

        self.sess = self.get_session('prediction')
        self.log.info('setting new weight_decay_rate={}'.format(weight_decay_rate))
        self.sess.run(self.model.assign_ops['weight_decay_rate_ow'], feed_dict={self.model.weight_decay_rate_ph: weight_decay_rate})
        self.log.info('Done restoring graph for global_step ({})'.format(self.global_step))

    def get_session(self, mode):
        """
        Returns a training/validation/prediction session.
        :param mode:  string of 'train'/'validation'/'prediction'
        :return: session or monitored session
        """
        lp = self.dataset.train_dataset.pool_size()
        if self.sess is None:
            # This should be the case only in the first time we call this function
            assert self.mode is None, 'sess in None but mode={}'.format(self.mode)
            self.log.info('Session is None at global_step={}.'.format(self.global_step))
        elif mode == self.mode:
            # do nothing
            return self.sess
        else:
            self.log.info('Closing current {} session at global_step={}'.format(self.mode, self.global_step))
            self.sess.close()

        self.log.info('Starting new {} session for global_step={}'.format(mode, self.global_step))
        if mode == 'train':
            sess = tf.train.MonitoredTrainingSession(
                checkpoint_dir=self.checkpoint_dir + '_' + str(lp),
                hooks=self.train_session_hooks,
                save_checkpoint_secs=self.checkpoint_secs,
                config=tf.ConfigProto(allow_soft_placement=True))
        elif mode == 'validation' or mode == 'prediction':
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir + '_' + str(lp)))
        else:
            err_str = 'mode {} is not expected in get_session()'.format(mode)
            self.log.error(err_str)
            raise AssertionError(err_str)

        self.log.info('DEBUG: global_step = {}. mode={}. prev_mode={}'.format(self.global_step, mode, self.mode))
        self.mode = mode
        return sess

    def build_train_env(self):
        self.log.info("Starting building the train environment")
        self.train_retention = Retention('retention_train', self.prm)  #TODO(gilad): Incorporate. Not in use
        self.summary_writer_train = tf.summary.FileWriter(self.train_dir)
        self.tb_logger_train = TBLogger(self.summary_writer_train)
        self.get_train_summaries()

        summary_hook = tf.train.SummarySaverHook(
            save_steps=self.summary_steps,
            summary_writer=self.summary_writer_train,
            summary_op=tf.summary.merge([self.model.summaries] + tf.get_collection(TRAIN_SUMMARIES))
        )
        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': self.model.global_step,
                     'loss_xent': self.model.xent_cost,
                     'loss_wd': self.model.wd_cost,
                     'loss': self.model.cost,
                     'score': self.model.score},
            every_n_iter=self.logger_steps)

        lp = self.dataset.train_dataset.pool_size()
        checkpoint_hook = GlobalStepCheckpointSaverHook(name='global_step_checkpoint_saver_hook',
                                                        prm=self.prm,
                                                        model=self.model,
                                                        steps_to_save=self.checkpoint_steps,
                                                        checkpoint_dir=self.checkpoint_dir + '_' + str(lp),
                                                        saver=self.saver,
                                                        checkpoint_basename='model_schedule.ckpt')

        self.train_session_hooks = [summary_hook, logging_hook, self.learning_rate_hook, checkpoint_hook]
