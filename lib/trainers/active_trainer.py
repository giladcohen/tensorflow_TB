from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.neighbors import NearestNeighbors
from lib.trainers.classification_trainer import ClassificationTrainer
from lib.active_kmean import KMeansWrapper
import numpy as np
from math import ceil
import os
from utils.misc import print_numpy


class ActiveTrainer(ClassificationTrainer):
    """Implementing active trainer
    Increasing the labeled pool gradually by using K-Means and K-NN
    Should run with DecayByScoreSetter
    """

    def __init__(self, *args, **kwargs):
        super(ActiveTrainer, self).__init__(*args, **kwargs)
        self.train_batch_count     = int(ceil(self.dataset.train_dataset.size / self.eval_batch_size))
        self.last_train_batch_size =          self.dataset.train_dataset.size % self.eval_batch_size
        self.min_learning_rate     = self.prm.train.train_control.MIN_LEARNING_RATE
        self.choice_of_new_labels  = self.prm.train.train_control.CHOICE_OF_NEW_LABELS

        self.clusters  = self.dataset.train_dataset.clusters
        self.cap       = self.dataset.train_dataset.cap
        self.num_fc_neurons = self.model.num_fc_neurons

        self.assert_config()

    def train_step(self):
        '''Implementing one training step'''
        lp = self.dataset.train_dataset.pool_size()

        if lp > self.cap:
            err_str = 'The train set pool size reached {} beyond the cap size ({})'.format(lp, self.cap)
            self.log.error(err_str)
            raise AssertionError(err_str)

        if self.learning_rate_hook.get_lrn_rate() >= self.min_learning_rate or lp == self.cap:
            # learning rate has not reached below the minimal value, or we reached the CAP - train normally
            super(ActiveTrainer, self).train_step()
            return

        # here we have learning rate <= min_learning_rate AND lp < CAP - need to choose next CLUSTERS labels
        # saving model at this stage:
        if self.debug_mode:
            self.log.info('Saving model_ref for global_step={} with pool size={}'.format(self.global_step, lp))
            checkpoint_file = os.path.join(self.checkpoint_dir, 'model_pool_{}.ckpt'.format(lp))
            pool_info_file  = os.path.join(self.root_dir, 'pool_info_{}'.format(lp))
            self.saver.save(self.get_session(self.sess),
                            checkpoint_file,
                            global_step=self.global_step)
            self.dataset.train_dataset.save_pool_data(pool_info_file)

        self.log.info('Adding {} new labels to train dataset using method: {}.'.format(self.clusters, self.choice_of_new_labels))
        if self.choice_of_new_labels == 'random':
            self.dataset.train_dataset.update_pool()
        elif self.choice_of_new_labels == 'kmeans':
            # analyzing (evaluation)
            features_vec = self.collect_train_features()

            # prediction
            KM = KMeansWrapper(name='KMeansWrapper', prm=self.prm, \
                               fixed_centers=features_vec[self.dataset.train_dataset.pool], \
                               n_clusters=lp + self.clusters, \
                               random_state=self.rand_gen)
            centers = KM.fit_predict_centers(features_vec)
            new_centers = centers[lp:(lp + self.clusters)]
            nbrs = NearestNeighbors(n_neighbors=1)
            nbrs.fit(features_vec)
            indices = nbrs.kneighbors(new_centers, return_distance=False)  # get indices of NNs of new centers
            indices = indices.T[0].tolist()

            # exclude existing labels in pool
            already_pooled_cnt = 0  # number of indices of samples that we added to pool already
            for myItem in indices:
                if myItem in self.dataset.train_dataset.pool:
                    already_pooled_cnt += 1
                    indices.remove(myItem)
                    self.log.info('Removing value {} from indices because it already exists in pool'.format(myItem))
            self.log.info('{} indices were already in pool. Randomized indices will be chosen instead of them'.format(already_pooled_cnt))
            self.dataset.train_dataset.update_pool(indices=indices)
            self.dataset.train_dataset.update_pool(clusters=already_pooled_cnt)
        else:
            err_str = 'Unfamiliar value of choice_of_new_labels ({}). Fix assert_config'.format(self.choice_of_new_labels)
            self.log.error(err_str)
            raise AssertionError(err_str)

        # reset learning rate to initial value
        self.learning_rate_hook.reset_learning_rate()
        self.retention.reset_memory()

    def collect_train_features(self):
        """Collecting all the features from the last layer (before the classifier) in the trainset"""
        features_vec = -1.0 * np.ones((self.dataset.train_dataset.size, self.num_fc_neurons), dtype=np.float32)
        total_samples = 0  # for debug
        self.log.info('start storing feature maps for the entire train set.')
        self.dataset.train_dataset.to_preprocess = False  # temporal setting
        for i in range(self.train_batch_count):
            b = i * self.eval_batch_size
            if i < (self.train_batch_count - 1) or (self.last_train_batch_size == 0):
                e = (i + 1) * self.eval_batch_size
            else:
                e = i * self.eval_batch_size + self.last_train_batch_size
            images, labels = self.dataset.get_mini_batch_train(indices=range(b, e))
            net = self.sess.run(self.model.net, feed_dict={self.model.images     : images,
                                                           self.model.labels     : labels,
                                                           self.model.is_training: False})
            features_vec[b:e] = np.reshape(net['pool_out'], (e - b, self.num_fc_neurons))
            total_samples += images.shape[0]
            self.log.info('Storing completed: {}%'.format(int(100.0 * e / self.dataset.train_dataset.size)))

            # debug
            features_tmp = np.array(features_vec[b:e])
            if np.sum(features_tmp == -1) == 0:
                err_str = 'feature_vec equals -1 for [b:e]=[{}:{}].'.format(b, e)
                print_numpy(features_tmp)
                self.log.error(err_str)
                raise AssertionError(err_str)
            
        assert total_samples == self.dataset.train_dataset.size, \
            'total_samples equals {} instead of {}'.format(total_samples, self.dataset.train_dataset.size)
        self.dataset.train_dataset.to_preprocess = True
        return features_vec

    def print_stats(self):
        super(ActiveTrainer, self).print_stats()
        self.log.info(' TRAIN_BATCH_COUNT: {}'.format(self.train_batch_count))
        self.log.info(' LAST_TRAIN_BATCH_SIZE: {}'.format(self.last_train_batch_size))
        self.log.info(' MIN_LEARNING_RATE: {}'.format(self.min_learning_rate))
        self.log.info(' CHOICE_OF_NEW_LABELS: {}'.format(self.choice_of_new_labels))

    def assert_config(self):
        if self.choice_of_new_labels != 'kmeans' and self.choice_of_new_labels != 'random':
            err_str = 'Unfamiliar value of choice_of_new_labels ({})'.format(self.choice_of_new_labels)
            self.log.error(err_str)
            raise AssertionError(err_str)
