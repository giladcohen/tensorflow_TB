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
import os


class DynamicModelTrainer(ActiveTrainerBase):

    def __init__(self, *args, **kwargs):
        super(DynamicModelTrainer, self).__init__(*args, **kwargs)
        self.checkpoint_dir = self.get_checkpoint_subdir()
        self.weight_decay_rate = self.prm.network.optimization.WEIGHT_DECAY_RATE

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
        """Updating the model - increasing the model parameters to accommodate larger pool"""
        tf.reset_default_graph()
        resnet_filters, self.weight_decay_rate, self.pca_embedding_dims = self.get_new_model_hps()
        self.checkpoint_dir = self.get_checkpoint_subdir()

        self.model = self.Factories.get_model()
        self.model.resnet_filters = resnet_filters
        self.embedding_dims = resnet_filters[-1]

        self.build()
        self.log.info('Done restoring graph for global_step ({})'.format(self.global_step))

    def get_new_model_hps(self):
        """
        :return: resnet filters, weight decay rate and the dims of the PCA embedding space
        """
        lp = self.dataset.train_dataset.pool_size()
        # [16, 160, 320, 640] for 50k samples.
        # for 1k: [16, 22, 44, 88]. weight_decay for 1k: 0.0390625. pca_embedding_dims = 18
        if lp == 2000:
            resnet_filters    = np.array([16, 32, 64, 128])
            weight_decay_rate = 0.01953125
            pca_embedding_dims = 26
        elif lp == 3000:
            resnet_filters = np.array([16, 40, 80, 160])
            weight_decay_rate = 0.01302083333
            pca_embedding_dims = 32
        elif lp == 4000:
            resnet_filters = np.array([16, 44, 88, 176])
            weight_decay_rate = 0.009765625
            pca_embedding_dims = 35
        elif lp == 5000:
            resnet_filters = np.array([16, 50, 100, 200])
            weight_decay_rate = 0.0078125
            pca_embedding_dims = 40
        else:
            err_str = 'pool size is {}. This is not possible'.format(lp)
            self.log.error(err_str)
            raise AssertionError(err_str)
        self.log.info('getting a new model for lp={}. resnet_filters={}. weight_decay_rate={}. pca_embedding_dims={}'
                      .format(lp, resnet_filters, weight_decay_rate, pca_embedding_dims))
        return resnet_filters, weight_decay_rate, pca_embedding_dims

    def get_train_summaries(self):
        super(DynamicModelTrainer, self).get_train_summaries()
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('weight_decay_rate', self.model.weight_decay_rate))
        # tf.add_to_collection(TRAIN_SUMMARIES, self.tb_logger_train.log_scalar('total_parameters', self.total_parameters, self.global_step))

    def finalize_graph(self):
        # overwrite the global step
        self.log.info('overwriting graph\'s global step to {}'.format(self.global_step))
        global_step_tensor = tf.train.get_or_create_global_step()
        update_global_step = global_step_tensor.assign(self.global_step)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(update_global_step)
        sess.close()

        self.sess = self.get_session('train')
        self.log.info('setting new weight_decay_rate={} with global_step={}'.format(self.weight_decay_rate, self.global_step))
        images, labels = self.dataset.get_mini_batch_train(indices=[0])
        self.sess.run([self.model.assign_ops['global_step_ow'], self.model.assign_ops['weight_decay_rate_ow']],
                      feed_dict={self.model.global_step_ph: self.global_step,
                                 self.model.weight_decay_rate_ph: self.weight_decay_rate,
                                 self.model.images: images,
                                 self.model.labels: labels,
                                 self.model.is_training: False})

    def set_params(self):
        pass

    def get_checkpoint_subdir(self):
        lp = self.dataset.train_dataset.pool_size()
        return os.path.join(self.prm.train.train_control.CHECKPOINT_DIR, 'checkpoint_' + str(lp))
