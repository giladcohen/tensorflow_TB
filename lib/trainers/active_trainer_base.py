from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.classification_trainer import ClassificationTrainer
from abc import ABCMeta, abstractmethod
import numpy as np
from math import ceil
import os
from sklearn.decomposition import PCA

class ActiveTrainerBase(ClassificationTrainer):
    """Implementing active trainer
    Increasing the labeled pool gradually by using K-Means and K-NN
    Should run with DecayByScoreSetter
    """
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(ActiveTrainerBase, self).__init__(*args, **kwargs)
        self.min_learning_rate          = self.prm.train.train_control.MIN_LEARNING_RATE
        self.annotation_rule            = self.prm.train.train_control.ANNOTATION_RULE
        self.steps_for_new_annotations  = self.prm.train.train_control.STEPS_FOR_NEW_ANNOTATIONS
        self.init_after_annot           = self.prm.train.train_control.INIT_AFTER_ANNOT

        self.pca_reduction = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims = self.prm.train.train_control.PCA_EMBEDDING_DIMS
        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)

        self._activate_annot = True

    def train(self):
        while not self.sess.should_stop():
            if self.to_annotate():
                self.annot_step()
                self._activate_annot = False
            elif self.to_eval():
                self.eval_step()
                self._activate_eval  = False
            else:
                self.train_step()
                self._activate_annot = True
                self._activate_eval  = True
        self.log.info('Stop training at global_step={}'.format(self.global_step))

    def annot_step(self):
        '''Implementing one annotation step'''
        self.log.info('Adding {} new labels to train dataset.'.format(self.dataset.clusters))
        new_indices = self.select_new_samples()  # select new indices
        self.add_new_samples(new_indices)        # add new indices to train dataset

        # reset learning rate to initial value, retention memory and model weights
        if self.init_after_annot:
            self.init_weights()
        self.learning_rate_hook.reset_learning_rate()
        self.validation_retention.reset_memory()

    @abstractmethod
    def select_new_samples(self):
        """
        Selecting new sampled to label
        :return: list of indices to add to train dataset
        """
        pass

    def add_new_samples(self, indices):
        """
        Adding indices to train dataset
        :param indices: list of indices to add to train dataset
        :return: no return
        """
        self.dataset.update_pool(indices=indices)
        self.dataset.set_handles(self.plain_sess)

    # def collect_features(self, dataset_type='train', dropout_keep_prob=1.0):
    #     """Collecting all the embedding features (before the classifier) and the prediction vectors in the dataset
    #     :param dataset_type: 'train' or 'validation'  #TODO(gilad): Support train/validation/test
    #     :return: feature vectors (embedding) and prediction vectors
    #     """
    #     if dataset_type == 'train':
    #         dataset = self.dataset.train_dataset
    #     elif dataset_type == 'validation':
    #         dataset = self.dataset.validation_dataset
    #     else:
    #         err_str = 'dataset_type={} is not supported'.format(dataset_type)
    #         self.log.error(err_str)
    #         raise AssertionError(err_str)
    #
    #     dataset.to_preprocess = False
    #     batch_count     = int(ceil(dataset.size / self.eval_batch_size))
    #     last_batch_size =          dataset.size % self.eval_batch_size
    #     features_vec    = -1.0 * np.ones((dataset.size, self.embedding_dims), dtype=np.float32)
    #     predictions_vec = -1.0 * np.ones((dataset.size, self.model.num_classes), dtype=np.float32)
    #     total_samples = 0  # for debug
    #
    #     self.log.info('start storing feature maps for the entire {} set.'.format(str(dataset)))
    #     for i in range(batch_count):
    #         b = i * self.eval_batch_size
    #         if i < (batch_count - 1) or (last_batch_size == 0):
    #             e = (i + 1) * self.eval_batch_size
    #         else:
    #             e = i * self.eval_batch_size + last_batch_size
    #         images, labels = dataset.get_mini_batch(indices=range(b, e))
    #         features, predictions = self.sess.run([self.model.net['embedding_layer'], self.model.predictions_prob],
    #                                                feed_dict={self.model.images           : images,
    #                                                           self.model.labels           : labels,
    #                                                           self.model.is_training      : False,
    #                                                           self.model.dropout_keep_prob: dropout_keep_prob})
    #         features_vec[b:e]    = np.reshape(features, (e - b, self.embedding_dims))
    #         predictions_vec[b:e] = np.reshape(predictions, (e - b, self.model.num_classes))
    #         total_samples += images.shape[0]
    #         self.log.info('Storing completed: {}%'.format(int(100.0 * e / dataset.size)))
    #
    #     assert total_samples == dataset.size, \
    #         'total_samples equals {} instead of {}'.format(total_samples, dataset.size)
    #     if dataset_type == 'train':
    #         dataset.to_preprocess = True
    #
    #     #FIXME(gilad): move pca transform after the collection of the features like in knn_classifier_tester
    #     if self.pca_reduction:
    #         self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.embedding_dims, self.pca_embedding_dims))
    #         if dataset_type == 'train':
    #             features_vec = self.pca.fit_transform(features_vec)
    #         else:
    #             features_vec = self.pca.transform(features_vec)
    #     return features_vec, predictions_vec

    def print_stats(self):
        super(ActiveTrainerBase, self).print_stats()
        self.log.info(' MIN_LEARNING_RATE: {}'.format(self.min_learning_rate))
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))
        self.log.info(' ANNOTATION_RULE: {}'.format(self.annotation_rule))
        self.log.info(' STEPS_FOR_NEW_ANNOTATIONS: {}'.format(self.steps_for_new_annotations))
        self.log.info(' INIT_AFTER_ANNOT: {}'.format(self.init_after_annot))

    def to_annotate(self):
        """
        :return: boolean. Whether or not to start an annotation phase
        """

        if not self._activate_annot:
            return False

        if self.annotation_rule == 'small_learning_rate':
            ret = self.learning_rate_hook.get_lrn_rate() < self.min_learning_rate and self.dataset.pool_size < self.dataset.cap
        elif self.annotation_rule == 'fixed_epochs':
            ret = self.global_step in self.steps_for_new_annotations
        else:
            err_str = 'annotation_rule={} is not supported'.format(self.annotation_rule)
            self.log.error(err_str)
            raise AssertionError(err_str)
        return ret

    def init_weights(self):
        self.log.info('Start initializing weights in global step={}'.format(self.global_step))
        self.plain_sess.run(self.model.init_op)
        self.log.info('Done initializing weights in global step={}'.format(self.global_step))

        # restore model global_step
        self.plain_sess.run(self.model.assign_ops['global_step_ow'],
                            feed_dict={self.model.global_step_ph: self.global_step})
        self.log.info('Done restoring global_step ({})'.format(self.global_step))

    def set_params(self):
        super(ActiveTrainerBase, self).set_params()
        assign_ops = []
        dropout_keep_prob = self.plain_sess.run(self.model.dropout_keep_prob)

        if not np.isclose(dropout_keep_prob, self.prm.network.system.DROPOUT_KEEP_PROB):
            assign_ops.append(self.model.assign_ops['dropout_keep_prob'])
            self.log.warning('changing model.dropout_keep_prob from {} to {}'.
                             format(dropout_keep_prob, self.prm.network.system.DROPOUT_KEEP_PROB))
        self.plain_sess.run(assign_ops)

    def train_step(self):
        '''Implementing one training step'''
        images, labels = self.dataset.get_mini_batch('train_pool', self.plain_sess)
        _ , self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
                                              feed_dict={self.model.images: images,
                                                         self.model.labels: labels,
                                                         self.model.is_training: True})
