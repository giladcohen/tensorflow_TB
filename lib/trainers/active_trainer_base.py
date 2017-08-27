from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.classification_trainer import ClassificationTrainer
from abc import ABCMeta, abstractmethod
import numpy as np
from math import ceil
import os
from utils.misc import print_numpy
from sklearn.decomposition import PCA


class ActiveTrainerBase(ClassificationTrainer):
    """Implementing active trainer
    Increasing the labeled pool gradually by using K-Means and K-NN
    Should run with DecayByScoreSetter
    """
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(ActiveTrainerBase, self).__init__(*args, **kwargs)
        self.min_learning_rate     = self.prm.train.train_control.MIN_LEARNING_RATE

        self.clusters  = self.dataset.train_dataset.clusters
        self.cap       = self.dataset.train_dataset.cap
        self.embedding_dims = self.model.embedding_dims
        self.pca_reduction = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims = self.prm.train.train_control.PCA_EMBEDDING_DIMS
        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)

    def train_step(self):
        '''Implementing one training step'''
        lp = self.dataset.train_dataset.pool_size()

        if lp > self.cap:
            err_str = 'The train set pool size reached {} beyond the cap size ({})'.format(lp, self.cap)
            self.log.error(err_str)
            raise AssertionError(err_str)

        if self.learning_rate_hook.get_lrn_rate() >= self.min_learning_rate or lp == self.cap:
            # learning rate has not reached below the minimal value, or we reached the CAP - train normally
            super(ActiveTrainerBase, self).train_step()
            return

        # here we have learning rate <= min_learning_rate AND lp < CAP - need to choose next CLUSTERS labels
        # saving model at this stage:
        self.debug_ops()
        self.log.info('Adding {} new labels to train dataset.'.format(self.clusters))
        new_indices = self.select_new_samples()  # select new indices
        self.add_new_samples(new_indices)        # add new indices to train dataset
        self.debug_ops()

        # reset learning rate to initial value
        self.learning_rate_hook.reset_learning_rate()
        self.retention.reset_memory()

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
        self.dataset.train_dataset.update_pool(indices=indices)

    def collect_features(self, dataset_type='train'):
        """Collecting all the embedding features (before the classifier) in the dataset
        :param dataset_type: 'train' or 'validation
        :return: no return
        """
        if dataset_type == 'train':
            dataset = self.dataset.train_dataset
        elif dataset_type == 'validation':
            dataset = self.dataset.validation_dataset
        else:
            err_str = 'dataset_type={} is not supported'.format(dataset_type)
            self.log.error(err_str)
            raise AssertionError(err_str)

        dataset.to_preprocess = False
        batch_count     = int(ceil(dataset.size / self.eval_batch_size))
        last_batch_size =          dataset.size % self.eval_batch_size
        features_vec    = -1.0 * np.ones((dataset.size, self.embedding_dims), dtype=np.float32)
        predictions_vec = -1.0 * np.ones((dataset.size, self.model.num_classes), dtype=np.float32)
        total_samples = 0  # for debug

        self.log.info('start storing feature maps for the entire {} set.'.format(str(dataset)))
        for i in range(batch_count):
            b = i * self.eval_batch_size
            if i < (batch_count - 1) or (last_batch_size == 0):
                e = (i + 1) * self.eval_batch_size
            else:
                e = i * self.eval_batch_size + last_batch_size
            images, labels = dataset.get_mini_batch(indices=range(b, e))
            net, predictions = self.sess.run([self.model.net, self.model.predictions_prob],
                                feed_dict={self.model.images     : images,
                                           self.model.labels     : labels,
                                           self.model.is_training: False})
            features_vec[b:e]    = np.reshape(net['embedding_layer'], (e - b, self.embedding_dims))
            predictions_vec[b:e] = np.reshape(predictions, (e - b, self.model.num_classes))
            total_samples += images.shape[0]
            self.log.info('Storing completed: {}%'.format(int(100.0 * e / dataset.size)))

        assert total_samples == dataset.size, \
            'total_samples equals {} instead of {}'.format(total_samples, dataset.size)
        if dataset_type == 'train':
            dataset.to_preprocess = True

        if self.pca_reduction:
            self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.embedding_dims, self.pca_embedding_dims))
            features_vec = self.pca.fit_transform(features_vec)
        return features_vec, predictions_vec

    def print_stats(self):
        super(ActiveTrainerBase, self).print_stats()
        self.log.info(' MIN_LEARNING_RATE: {}'.format(self.min_learning_rate))
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))


    def debug_ops(self):
        lp = self.dataset.train_dataset.pool_size()
        if self.debug_mode:
            self.log.info('Saving model_ref for global_step={} with pool size={}'.format(self.global_step, lp))
            checkpoint_file = os.path.join(self.checkpoint_dir, 'model_pool_{}.ckpt'.format(lp))
            pool_info_file = os.path.join(self.root_dir, 'pool_info_{}'.format(lp))
            self.saver.save(self.get_session(self.sess),
                            checkpoint_file,
                            global_step=self.global_step)
            self.dataset.train_dataset.save_pool_data(pool_info_file)
