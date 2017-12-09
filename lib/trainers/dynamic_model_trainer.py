from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.active_trainer_base import ActiveTrainerBase
import numpy as np
import tensorflow as tf
from lib.base.collections import TRAIN_SUMMARIES
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
                if self.sess.should_stop():
                    self.log.info('Stop training at global_step={}'.format(self.global_step))
                    break

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

        images, labels = self.dataset.get_mini_batch_train(indices=[0])
        feed_dict = {self.model.global_step_ph      : self.global_step,
                     self.model.weight_decay_rate_ph: self.weight_decay_rate,
                     self.model.images              : images,
                     self.model.labels              : labels,
                     self.model.is_training         : False}

        self.sess = self.get_session('train')
        self.sess.run([self.model.assign_ops['global_step_ow'], self.model.assign_ops['weight_decay_rate_ow']], feed_dict=feed_dict)
        global_step, weight_decay_rate = self.sess.run([self.model.global_step, self.model.weight_decay_rate], feed_dict=feed_dict)

        if global_step != self.global_step:
            err_str = 'returned global_step={} is different than self.global_step={}'.format(global_step, self.global_step)
            self.log.error(err_str)
            raise AssertionError(err_str)
        if not np.isclose(weight_decay_rate, self.weight_decay_rate):
            err_str = 'returned weight_decay_rate={} is different than self.weight_decay_rate={}'.format(weight_decay_rate, self.weight_decay_rate)
            self.log.error(err_str)
            raise AssertionError(err_str)

    def set_params(self):
        pass

    def get_checkpoint_subdir(self):
        lp = self.dataset.train_dataset.pool_size()
        return os.path.join(self.prm.train.train_control.CHECKPOINT_DIR, 'checkpoint_' + str(lp))
