from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.classification_trainer import ClassificationTrainer
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
from utils.misc import collect_features
from sklearn.neighbors import KNeighborsClassifier

class SemiSupervisedTrainer(ClassificationTrainer):
    """Implementing active trainer
    Increasing the labeled pool gradually by using K-Means and K-NN
    Should run with DecayByScoreSetter
    """

    def __init__(self, *args, **kwargs):
        super(SemiSupervisedTrainer, self).__init__(*args, **kwargs)
        self.soft_label_update_steps    = self.prm.train.train_control.semi_supervised.SOFT_LABEL_UPDATE_STEPS

        self.pca_reduction = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims = self.prm.train.train_control.PCA_EMBEDDING_DIMS
        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)

        self._activate_sl_update = False
        self._finalized_once = False
        self._num_of_updates = 0

    def train(self):
        while not self.sess.should_stop():
            if self.to_update():
                self.update_soft_labels()
                self.update_graph()
                self._activate_sl_update = False
            elif self.to_eval():
                self.eval_step()
                self._activate_eval  = False
            elif self.to_test():
                self.test_step()
                self._activate_test = False
            else:
                self.train_step()
                self._activate_sl_update = True
                self._activate_eval  = True
                self._activate_test  = True
        self.log.info('Stop training at global_step={}'.format(self.global_step))

    def update_soft_labels(self):
        """ Updating the dataset wrapper with the new soft labels for the unpool_train dataset
        :return: None
        """
        self.log.info('Getting embedding space vectors for all samples in the train_pool dataset')
        pool_features_vec, pool_labels = \
            collect_features(agent=self,
                             dataset_name='train_pool_eval',
                             fetches=[self.model.net['embedding_layer'], self.model.labels],
                             feed_dict={self.model.dropout_keep_prob: 1.0})
        pool_labels = np.argmax(pool_labels, axis=1)

        self.log.info('Getting embedding space vectors for all samples in the train_unpool dataset')
        (unpool_features_vec,) = \
            collect_features(agent=self,
                             dataset_name='train_unpool_eval',
                             fetches=[self.model.net['embedding_layer']],
                             feed_dict={self.model.dropout_keep_prob: 1.0})

        self.log.info('building kNN space only for the labeled (pooled) train features')
        nbrs = KNeighborsClassifier(n_neighbors=30, weights='uniform', p=1)
        nbrs.fit(pool_features_vec, pool_labels)

        self.log.info('Calculating the estimated labels probability based on KNN')
        train_unpool_soft_labels = nbrs.predict_proba(unpool_features_vec)

        self.dataset.update_soft_labels(train_unpool_soft_labels, self.global_step)
        self._num_of_updates += 1

    def update_graph(self):
        """Resetting the graph and starting a new graph to update the dataset operations on the graph"""
        self.sess.close()
        tf.reset_default_graph()
        train_validation_map_ref = self.dataset.train_validation_map_ref
        train_unpool_soft_labels = self.dataset.train_unpool_soft_labels

        self.model   = self.Factories.get_model()
        self.dataset = self.Factories.get_dataset()

        self.dataset.train_validation_map_ref = train_validation_map_ref
        self.dataset.train_unpool_soft_labels = train_unpool_soft_labels

        self.build()
        self.log.info('Done restoring graph for global_step ({})'.format(self.global_step))

    def finalize_graph(self):
        """overwrite the global step on the graph"""
        if self._finalized_once:
            self.log.info('overwriting graph\'s value: global_step={}'.format(self.global_step))
            self.plain_sess.run(self.model.assign_ops['global_step_ow'], feed_dict={self.model.global_step_ph: self.global_step})
            self.dataset.set_handles(self.plain_sess)
        else:
            # restoring global_step
            super(SemiSupervisedTrainer, self).finalize_graph()
            self._finalized_once = True

    def print_stats(self):
        super(SemiSupervisedTrainer, self).print_stats()
        self.log.info(' SOFT_LABEL_UPDATE_STEPS: {}'.format(self.soft_label_update_steps))
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))

    def to_update(self):
        """
        :return: boolean. Whether or not to update the dataset with the new KNN model predictions
        """
        return self.global_step % self.soft_label_update_steps == 0 and self._activate_sl_update

    def train_step(self):
        '''Implementing one training step'''
        if self._num_of_updates == 0:
            _, images, labels = self.dataset.get_mini_batch('train_pool_init', self.plain_sess)
        else:
            _, pool_images  , pool_labels   = self.dataset.get_mini_batch('train_pool'  , self.plain_sess)
            _, unpool_images, unpool_labels = self.dataset.get_mini_batch('train_unpool', self.plain_sess)
            images = np.concatenate((pool_images, unpool_images), axis=0)
            labels = np.concatenate((pool_labels, unpool_labels), axis=0)

        _ , self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
                                              feed_dict={self.model.images: images,
                                                         self.model.labels: labels,
                                                         self.model.is_training: True})
