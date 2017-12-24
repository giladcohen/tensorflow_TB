from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.base.collections import TRAIN_SUMMARIES
from lib.trainers.classification_trainer import ClassificationTrainer
from math import ceil
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score


class DMLClassificationTrainer(ClassificationTrainer):
    """Implementing classification trainer for DML
    Using the entire labeled trainset for training"""

    def __init__(self, *args, **kwargs):
        super(DMLClassificationTrainer, self).__init__(*args, **kwargs)
        self.pca_reduction = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims = self.prm.train.train_control.PCA_EMBEDDING_DIMS
        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)
        self.knn = KNeighborsClassifier(n_neighbors=30, p=2)

    def print_stats(self):
        super(DMLClassificationTrainer, self).print_stats()
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))

    def eval_step(self):
        '''Implementing one evaluation step.'''
        self.log.info('start running eval within training. global_step={}'.format(self.global_step))
        train_size       = 5000 #self.dataset.train_dataset.pool_size()
        validation_size  = 500  #self.dataset.validation_dataset.size

        X_train    = self.collect_features(dataset_type='train')
        _, y_train = self.dataset.get_mini_batch_train(indices=range(train_size))

        X_test    = self.collect_features(dataset_type='validation')
        _, y_test = self.dataset.get_mini_batch_validate(indices=range(validation_size))

        self.log.info('Fitting KNN model...')
        self.knn.fit(X_train, y_train)

        self.log.info('Predicting test set labels from KNN model...')
        y_pred = self.knn.predict(X_test)
        score     = np.sum(y_pred==y_test)/validation_size
        nmi_score = normalized_mutual_info_score(labels_true=y_test, labels_pred=y_pred)
        summaries, loss = self.sample_eval_stats()
        self.validation_retention.add_score(score, self.global_step)

        self.tb_logger_eval.log_scalar('score', score, self.global_step)
        self.tb_logger_eval.log_scalar('best score', self.validation_retention.get_best_score(), self.global_step)
        self.tb_logger_eval.log_scalar('nmi_score', nmi_score, self.global_step)

        self.summary_writer_eval.add_summary(summaries, self.global_step)
        self.summary_writer_eval.flush()
        self.log.info('EVALUATION (step={}): loss: {}, score: {}, nmi_score: {}, best score: {}' \
                      .format(self.global_step, loss, score, nmi_score, self.validation_retention.get_best_score()))

    def sample_eval_stats(self):
        """Sampling validation summary and loss only for one eval batch."""
        images, labels = self.dataset.get_mini_batch_validate(indices=range(self.eval_batch_size))
        (summaries, loss) = self.sess.run([self.model.summaries, self.model.cost],
                                          feed_dict={self.model.images: images,
                                                     self.model.labels: labels,
                                                     self.model.is_training: False})
        return summaries, loss

    def collect_features(self, dataset_type, dropout_keep_prob=1.0):
        """Collecting all the embedding features in the dataset
        :param dataset_type: 'train' or 'validation'
        :return: feature vectors (embedding)
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
        features_vec    = -1.0 * np.ones((dataset.size, self.model.embedding_dims), dtype=np.float32)
        total_samples = 0  # for debug

        self.log.info('start storing feature maps for the entire {} set.'.format(str(dataset)))
        for i in range(batch_count):
            b = i * self.eval_batch_size
            if i < (batch_count - 1) or (last_batch_size == 0):
                e = (i + 1) * self.eval_batch_size
            else:
                e = i * self.eval_batch_size + last_batch_size
            images, labels = dataset.get_mini_batch(indices=range(b, e))
            features = self.sess.run(self.model.net['embedding_layer'],
                                     feed_dict={self.model.images           : images,
                                                self.model.labels           : labels,
                                                self.model.is_training      : False,
                                                self.model.dropout_keep_prob: dropout_keep_prob})
            features_vec[b:e]    = np.reshape(features, (e - b, self.model.embedding_dims))
            total_samples += images.shape[0]
            self.log.info('Storing completed: {}%'.format(int(100.0 * e / dataset.size)))

        assert total_samples == dataset.size, 'total_samples equals {} instead of {}'.format(total_samples, dataset.size)
        if dataset_type == 'train':
            dataset.to_preprocess = True

        # FIXME(gilad): move pca transform after the collection of the features like in knn_classifier_tester
        if self.pca_reduction:
            self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.model.embedding_dims, self.pca_embedding_dims))
            if dataset_type == 'train':
                features_vec = self.pca.fit_transform(features_vec)
            else:
                features_vec = self.pca.transform(features_vec)

        return features_vec

