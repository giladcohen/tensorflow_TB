from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer import ActiveTrainer
import numpy as np
import tensorflow as tf
from lib.base.collections import TRAIN_SUMMARIES
import os


class DynamicModelTrainer(ActiveTrainer):

    def __init__(self, *args, **kwargs):
        super(DynamicModelTrainer, self).__init__(*args, **kwargs)
        self.checkpoint_dir = self.get_checkpoint_subdir()
        self.weight_decay_rate = self.prm.network.optimization.WEIGHT_DECAY_RATE

    def update_graph(self):
        """Updating the model - increasing the model parameters to accommodate larger pool"""
        tf.reset_default_graph()
        resnet_filters, self.weight_decay_rate, self.pca_embedding_dims = self.get_new_model_hps()
        self.checkpoint_dir = self.get_checkpoint_subdir()
        train_validation_map_ref = self.dataset.train_validation_map_ref

        self.model = self.Factories.get_model()
        self.model.resnet_filters = resnet_filters
        self.dataset = self.Factories.get_dataset()
        self.dataset.train_validation_map_ref = train_validation_map_ref

        self.build()
        self.log.info('Done restoring graph for global_step ({})'.format(self.global_step))

    def get_new_model_hps(self):
        """
        :return: resnet filters, weight decay rate and the dims of the PCA embedding space
        """
        lp = self.dataset.train_dataset.pool_size
        # [16, 160, 320, 640] for 50k samples.
        # for 1k: [16, 22, 44, 88]. weight_decay for 1k: 0.0390625. pca_embedding_dims = 18
        if lp == 2000:
            resnet_filters    = np.array([16, 32, 64, 128])
            weight_decay_rate = 0.007
            pca_embedding_dims = 26
        elif lp == 3000:
            resnet_filters = np.array([16, 40, 80, 160])
            weight_decay_rate = 0.004
            pca_embedding_dims = 32
        elif lp == 4000:
            resnet_filters = np.array([16, 44, 88, 176])
            weight_decay_rate = 0.0035
            pca_embedding_dims = 35
        elif lp == 5000:
            resnet_filters = np.array([16, 50, 100, 200])
            weight_decay_rate = 0.0026
            pca_embedding_dims = 40
        else:
            err_str = 'pool size is {}. This is not possible'.format(lp)
            self.log.error(err_str)
            raise AssertionError(err_str)
        self.log.info('getting a new model for lp={}. resnet_filters={}. weight_decay_rate={}. pca_embedding_dims={}'
                      .format(lp, resnet_filters, weight_decay_rate, pca_embedding_dims))
        return resnet_filters, weight_decay_rate, pca_embedding_dims

    def finalize_graph(self):
        # overwrite the global step and weight decay rate
        self.log.info('overwriting graph\'s values: global_step={}, weight_decay_rate={}'
                      .format(self.global_step, self.weight_decay_rate))
        self.sess.run([self.model.assign_ops['global_step_ow'], self.model.assign_ops['weight_decay_rate_ow']],
                      feed_dict={self.model.global_step_ph       : self.global_step,
                                 self.model.weight_decay_rate_ph : self.weight_decay_rate})
        self.dataset.set_handles(self.plain_sess)

    def set_params(self):
        pass

    def get_checkpoint_subdir(self):
        lp = self.dataset.train_dataset.pool_size
        return os.path.join(self.prm.train.train_control.CHECKPOINT_DIR, 'checkpoint_' + str(lp))
