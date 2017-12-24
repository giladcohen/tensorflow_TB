from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from lib.base.agent_base import AgentBase
from utils.tensorboard_logging import TBLogger
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import normalized_mutual_info_score
from math import ceil

class KNNClassifierTester(AgentBase):

    def __init__(self, name, prm, model, dataset):
        super(KNNClassifierTester, self).__init__(name)
        self.prm     = prm
        self.model   = model
        self.dataset = dataset

        self.rand_gen = np.random.RandomState(self.prm.SUPERSEED)
        self.debug_mode = self.prm.DEBUG_MODE

        self.eval_batch_size       = self.prm.train.train_control.EVAL_BATCH_SIZE
        self.root_dir              = self.prm.train.train_control.ROOT_DIR
        self.pred_dir              = self.prm.train.train_control.PREDICTION_DIR
        self.checkpoint_dir        = self.prm.train.train_control.CHECKPOINT_DIR
        self.pca_reduction         = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims    = self.prm.train.train_control.PCA_EMBEDDING_DIMS

        # testing parameters
        self.tester          = self.prm.test.test_control.TESTER         # just used for printing.
        self.checkpoint_file = self.prm.test.test_control.CHECKPOINT_FILE
        self.knn_neighbors   = self.prm.test.test_control.KNN_NEIGHBORS
        self.knn_p_norm      = self.prm.test.test_control.KNN_P_NORM
        self.knn_jobs        = self.prm.test.test_control.KNN_JOBS
        self.dump_net        = self.prm.test.test_control.DUMP_NET

        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)
        self.knn = KNeighborsClassifier(n_neighbors=self.knn_neighbors, p=self.knn_p_norm, n_jobs=self.knn_jobs)

    def build(self):
        """
        Building all tester agents: test session
        """
        self.model.build_graph()
        # self.print_model_info()
        self.saver = tf.train.Saver(max_to_keep=None, name='test', filename='model_pred')
        self.build_prediction_env()

        # create session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # restore checkpoint
        self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, self.checkpoint_file))

        self.log.info('Done building tester {}'.format(str(self)))

    def test(self):
        train_size = self.dataset.train_dataset.pool_size()
        test_size  = self.dataset.validation_dataset.size

        self.log.info('Collecting {} train set embedding features'.format(train_size))
        X_train_features = \
            self.collect_features(
                fetches=[self.model.net['embedding_layer']],
                dataset_type='train'
            )
        _, y_train = self.dataset.get_mini_batch_train(indices=range(train_size))

        self.log.info('Collecting {} test set embedding features and DNN predictions'.format(test_size))
        X_test_features, X_test_dnn_predictions_prob = \
            self.collect_features(
                fetches=[self.model.net['embedding_layer'], self.model.predictions_prob],
                dataset_type='validation'
            )
        _, y_test = self.dataset.get_mini_batch_train(indices=range(test_size))

        if self.pca_reduction:
            self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.model.embedding_dims, self.pca_embedding_dims))
            X_train_features_post = self.pca.fit_transform(X_train_features)
            X_test_features_post  = self.pca.transform(X_test_features)
        else:
            X_train_features_post = X_train_features
            X_test_features_post  = X_test_features

        self.log.info('Fitting KNN model...')
        self.knn.fit(X_train_features_post, y_train)

        self.log.info('Predicting test set labels from KNN model...')
        y_pred = self.knn.predict(X_test_features_post)
        score     = np.sum(y_pred==y_test)/test_size
        nmi_score = normalized_mutual_info_score(labels_true=y_test, labels_pred=y_pred)

        self.tb_logger_pred.log_scalar('score', score, 0)
        self.tb_logger_pred.log_scalar('NMI score', nmi_score, 0)

        self.summary_writer_pred.flush()
        self.log.info('TEST : score: {}, NMI score: {}'.format(score, nmi_score))

        if self.dump_net:
            train_features_file            = os.path.join(self.pred_dir, 'train_features.npy')
            test_features_file             = os.path.join(self.pred_dir, 'test_features.npy')
            test_dnn_predictions_prob_file = os.path.join(self.pred_dir, 'test_predictions_prob.npy')

            self.log.info('Dumping train features into disk:\n{}\n{}\n{})'
                          .format(train_features_file, test_features_file, test_dnn_predictions_prob_file))
            np.save(train_features_file           , X_train_features)
            np.save(X_test_features               , X_test_features)
            np.save(test_dnn_predictions_prob_file, X_test_dnn_predictions_prob)

        self.log.info('Tester {} is done'.format(str(self)))

    def print_model_info(self):
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        self.total_parameters = param_stats.total_parameters
        self.log.info('total_params: {}\n'.format(self.total_parameters))

        tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    def build_prediction_env(self):
        self.log.info("Starting building the prediction environment")
        self.summary_writer_pred = tf.summary.FileWriter(self.pred_dir)
        self.tb_logger_pred = TBLogger(self.summary_writer_pred)

    def print_stats(self):
        '''print basic test parameters'''
        self.log.info('Test parameters:')
        self.log.info(' DEBUG_MODE: {}'.format(self.debug_mode))
        self.log.info(' EVAL_BATCH_SIZE: {}'.format(self.eval_batch_size))
        self.log.info(' ROOT_DIR: {}'.format(self.root_dir))
        self.log.info(' PREDICTION_DIR: {}'.format(self.pred_dir))
        self.log.info(' CHECKPOINT_DIR: {}'.format(self.checkpoint_dir))
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))
        self.log.info(' TESTER: {}'.format(self.tester))
        self.log.info(' CHECKPOINT_FILE: {}'.format(self.checkpoint_file))
        self.log.info(' KNN_NEIGHBORS: {}'.format(self.knn_neighbors))
        self.log.info(' KNN_P_NORM: {}'.format(self.knn_p_norm))
        self.log.info(' KNN_JOBS: {}'.format(self.knn_jobs))
        self.log.info(' DUMP_NET: {}'.format(self.dump_net))

    def collect_features(self, fetches, dataset_type, dropout_keep_prob=1.0):
        """Collecting all fetches from the DNN in the dataset
        :param fetches: list of all the fetches to sample from the DNN.
        :param dataset_type: 'train' or 'validation'
        :return: feature vectors (embedding)
        """

        # Assuming all fetches are 1D. Otherwise code needs to be rewritten
        fetches_dims = [fetches[i].get_shape().as_list()[-1] for i in xrange(len(fetches))]

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
        fetches_np = [-1.0 * np.ones((dataset.size, fetches_dims[i]), dtype=np.float32) for i in xrange(len(fetches))]
        total_samples = 0  # for debug

        self.log.info('start storing feature maps for the entire {} set.'.format(str(dataset)))
        for i in range(batch_count):
            b = i * self.eval_batch_size
            if i < (batch_count - 1) or (last_batch_size == 0):
                e = (i + 1) * self.eval_batch_size
            else:
                e = i * self.eval_batch_size + last_batch_size
            images, labels = dataset.get_mini_batch(indices=range(b, e))
            fetches_out = self.sess.run(fetches,
                                        feed_dict={self.model.images           : images,
                                                   self.model.labels           : labels,
                                                   self.model.is_training      : False,
                                                   self.model.dropout_keep_prob: dropout_keep_prob})
            for i in xrange(len(fetches)):
                fetches_np[i][b:e] = np.reshape(fetches_out[i], (e - b, fetches_dims[i]))
            total_samples += images.shape[0]
            self.log.info('Storing completed: {}%'.format(int(100.0 * e / dataset.size)))

        assert total_samples == dataset.size, 'total_samples equals {} instead of {}'.format(total_samples, dataset.size)
        if dataset_type == 'train':
            dataset.to_preprocess = True

        return fetches_np
