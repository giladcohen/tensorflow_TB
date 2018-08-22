from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from lib.testers.tester_base import TesterBase
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from utils.misc import collect_features, calc_mutual_agreement, calc_psame
from scipy.stats import entropy

eps = 0.000001

class KNNClassifierTester(TesterBase):

    def __init__(self, *args, **kwargs):
        super(KNNClassifierTester, self).__init__(*args, **kwargs)
        self.decision_method = self.prm.test.test_control.DECISION_METHOD

        self.pca_reduction         = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims    = self.prm.train.train_control.PCA_EMBEDDING_DIMS

        # testing parameters
        self.knn_neighbors   = self.prm.test.test_control.KNN_NEIGHBORS
        self.knn_norm        = self.prm.test.test_control.KNN_NORM
        self.knn_weights     = self.prm.test.test_control.KNN_WEIGHTS
        self.knn_jobs        = self.prm.test.test_control.KNN_JOBS

        self.num_classes     = self.dataset.num_classes

        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)
        self.tested_layer       = 'embedding_layer'
        self.eval_trainset      = self.prm.test.test_control.EVAL_TRAINSET
        self.randomized_dataset = 'random' in str(self.dataset)

        if self.knn_norm not in ['L1', 'L2']:
            err_str = 'knn_norm {} is not supported'.format(self.knn_norm)
            self.log.error(err_str)
            raise AssertionError(err_str)

        self.knn = KNeighborsClassifier(
            n_neighbors=self.knn_neighbors,
            weights=self.knn_weights,
            p=int(self.knn_norm[-1]),
            n_jobs=self.knn_jobs)

        self.knn_train = KNeighborsClassifier(
            n_neighbors=self.knn_neighbors + 1,
            weights=self.knn_weights,
            p=int(self.knn_norm[-1]),
            n_jobs=self.knn_jobs)

        self.svm = SVC(
            kernel='linear',
            probability=True,
            random_state=self.rand_gen)

        self.lr = LogisticRegression(
            penalty=self.knn_norm.lower(),
            dual=False,
            random_state=self.rand_gen,
            n_jobs=self.knn_jobs
        )

    def fetch_dump_data_features(self, layer_name=None, test_dir=None):
        """Optionally fetching precomputed train/test features, and labels."""
        if layer_name is None:
            layer_name = self.tested_layer
        if test_dir is None:
            test_dir = self.test_dir
        train_features_file             = os.path.join(test_dir, 'train_features.npy')
        test_features_file              = os.path.join(test_dir, 'test_features.npy')
        train_dnn_predictions_prob_file = os.path.join(test_dir, 'train_dnn_predictions_prob.npy')
        test_dnn_predictions_prob_file  = os.path.join(test_dir, 'test_dnn_predictions_prob.npy')
        train_labels_file               = os.path.join(test_dir, 'train_labels.npy')
        test_labels_file                = os.path.join(test_dir, 'test_labels.npy')

        if self.load_from_disk:
            self.log.info('Loading {}/{} train/test set embedding features from disk'.format(self.dataset.train_set_size, self.dataset.test_set_size))
            X_train_features           = np.load(train_features_file)
            y_train                    = np.load(train_labels_file)
            train_dnn_predictions_prob = np.load(train_dnn_predictions_prob_file)
            X_test_features            = np.load(test_features_file)
            y_test                     = np.load(test_labels_file)
            test_dnn_predictions_prob  = np.load(test_dnn_predictions_prob_file)
        else:
            if self.randomized_dataset:
                dataset_name = 'train_random_eval'
            else:
                dataset_name = 'train_eval'
            self.log.info('Collecting {} samples for training from layer: {} from dataset: {}'.format(self.dataset.train_set_size, layer_name, dataset_name))
            (X_train_features, y_train, train_dnn_predictions_prob) = \
                collect_features(
                    agent=self,
                    dataset_name=dataset_name,
                    fetches=[self.model.net[layer_name], self.model.labels, self.model.predictions_prob],
                    feed_dict={self.model.dropout_keep_prob: 1.0})
            dataset_name = 'test'
            self.log.info('Collecting {} samples for testing from layer: {} from dataset: {}'.format(self.dataset.test_set_size, layer_name, dataset_name))
            (X_test_features, y_test, test_dnn_predictions_prob) = \
                collect_features(
                    agent=self,
                    dataset_name=dataset_name,
                    fetches=[self.model.net[layer_name], self.model.labels, self.model.predictions_prob],
                    feed_dict={self.model.dropout_keep_prob: 1.0})

        if self.dump_net:
            self.log.info('Dumping train features into disk:\n{}\n{}\n{}\n{}\n{}'
                          .format(train_features_file, test_features_file, test_dnn_predictions_prob_file, train_labels_file, test_labels_file))
            np.save(train_features_file            , X_train_features)
            np.save(test_features_file             , X_test_features)
            np.save(train_dnn_predictions_prob_file, train_dnn_predictions_prob)
            np.save(test_dnn_predictions_prob_file , test_dnn_predictions_prob)
            np.save(train_labels_file              , y_train)
            np.save(test_labels_file               , y_test)

        return X_train_features, X_test_features, train_dnn_predictions_prob, test_dnn_predictions_prob, y_train, y_test

    def apply_pca(self, X, fit=False):
        """If pca_reduction is True, apply PCA reduction"""
        if self.pca_reduction:
            self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.model.embedding_dims, self.pca_embedding_dims))
            if fit:
                self.pca.fit(X)
            X = self.pca.transform(X)
        return X

    def knn_predict_proba_for_trainset(self, model, X_train_features, y_train):
        """
        :param model: knn_train
        :param X_train_features: Training set features ([n_samples, n_features])
        :param y_train: training set gt ([n_samples])
        :return: knn predictions of the training set, using an efficient leave-one-out.
        """
        biased_knn_predictions_prob_train = model.predict_proba(X_train_features)
        knn_predictions_prob_train = np.zeros(biased_knn_predictions_prob_train.shape)

        for i in range(len(X_train_features)):
            y = int(y_train[i])
            proba = biased_knn_predictions_prob_train[i]
            assert proba[y] >= 1/(self.knn_neighbors + 1), "for i={}: prob[y={}] = {}, but cannot be smaller than {}"\
                .format(i, y, proba[y], 1/(self.knn_neighbors + 1))
            proba[y] -= 1/(self.knn_neighbors + 1)
            proba *= (self.knn_neighbors + 1)/self.knn_neighbors
            assert np.isclose(sum(proba), 1.0), "sum of proba[i={}] is {} instead of 1.0".format(i, sum(proba))
            knn_predictions_prob_train[i] = proba

        return knn_predictions_prob_train

    def test(self):
        X_train_features, \
        X_test_features, \
        train_dnn_predictions_prob, \
        test_dnn_predictions_prob, \
        y_train, \
        y_test = self.fetch_dump_data_features()

        X_train_features = self.apply_pca(X_train_features, fit=True)
        X_test_features  = self.apply_pca(X_test_features , fit=False)

        if 'knn' in self.decision_method:
            self.log.info('Fitting KNN model...')
            self.knn.fit(X_train_features, y_train)
            self.knn_train.fit(X_train_features, y_train)
        if 'svm' in self.decision_method:
            self.log.info('Fitting SVM model...')
            self.svm.fit(X_train_features, y_train)
        if 'logistic_regression' in self.decision_method:
            self.log.info('Fitting Logistic Regression model...')
            self.lr.fit(X_train_features, y_train)

        if self.decision_method == 'dnn_accuracy':
            y_pred = test_dnn_predictions_prob.argmax(axis=1)
        elif self.decision_method == 'knn_accuracy':
            self.log.info('Predicting test set labels from KNN model...')
            test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
            y_pred = test_knn_predictions_prob.argmax(axis=1)
        elif self.decision_method == 'svm':
            self.log.info('Predicting test set labels from SVM model...')
            y_pred = self.svm.predict(X_test_features)
        elif self.decision_method == 'logistic_regression':
            self.log.info('Predicting test set labels from Logistic Regression model...')
            y_pred = self.lr.predict(X_test_features)
        elif self.decision_method == 'dnn_svm_psame':
            self.log.info('Predicting labels from DNN model...')
            y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
            self.log.info('Predicting labels from SVM model...')
            y_pred_svm = self.svm.predict(X_test_features)
            psame = calc_psame(y_pred_dnn, y_pred_svm)

            score_str = 'score_metrics/layer={}/decision_method={}/kernel=rbf/norm={}/PCA={}' \
                .format(self.tested_layer, self.decision_method, self.knn_norm, self.pca_embedding_dims)
            self.tb_logger_test.log_scalar(score_str, psame, self.global_step)
            print_str = '{}: psame={}.'.format(score_str, psame)
            self.log.info(print_str)
            print(print_str)
            self.summary_writer_test.flush()
            exit(0)
        elif self.decision_method == 'dnn_knn_psame':
            self.log.info('Predicting labels from DNN model...')
            y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
            self.log.info('Predicting labels from KNN model...')
            test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
            y_pred_knn = test_knn_predictions_prob.argmax(axis=1)
            psame = calc_psame(y_pred_dnn, y_pred_knn)

            score_str = 'score_metrics/layer={}/decision_method={}/kernel=rbf/norm={}/PCA={}' \
                .format(self.tested_layer, self.decision_method, self.knn_norm, self.pca_embedding_dims)
            self.tb_logger_test.log_scalar(score_str, psame, self.global_step)
            print_str = '{}: psame={}.'.format(score_str, psame)
            self.log.info(print_str)
            print(print_str)
            self.summary_writer_test.flush()
            exit(0)
        elif self.decision_method == 'dnn_logistic_regression_psame':
            self.log.info('Predicting labels from DNN model...')
            y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
            self.log.info('Predicting labels from Logistic Regression model...')
            y_pred_lr = self.lr.predict(X_test_features)
            psame = calc_psame(y_pred_dnn, y_pred_lr)

            score_str = 'score_metrics/layer={}/decision_method={}/norm={}/PCA={}' \
                .format(self.tested_layer, self.decision_method, self.knn_norm, self.pca_embedding_dims)
            self.tb_logger_test.log_scalar(score_str, psame, self.global_step)
            print_str = '{}: psame={}.'.format(score_str, psame)
            self.log.info(print_str)
            print(print_str)
            self.summary_writer_test.flush()
            exit(0)
        elif self.decision_method == 'knn_svm_logistic_regression_metrics':
            self.log.info('Predicting labels from SVM model...')
            y_prob_svm = self.svm.predict_proba(X_test_features)
            y_pred_svm = y_prob_svm.argmax(axis=1)
            svm_score = np.average(y_test == y_pred_svm)
            self.log.info('Predicting labels from Logistic Regression model...')
            y_prob_lr = self.lr.predict_proba(X_test_features)
            y_pred_lr = y_prob_lr.argmax(axis=1)
            lr_score = np.average(y_test == y_pred_lr)
            self.log.info('Predicting labels from KNN model...')
            y_prob_knn = self.knn.predict_proba(X_test_features)
            y_pred_knn = y_prob_knn.argmax(axis=1)
            knn_score = np.average(y_test == y_pred_knn)

            self.log.info('Predicting SVM||KNN PSAME...')
            svm_knn_psame = calc_psame(y_pred_svm, y_pred_knn)
            self.log.info('Predicting SVM||LR PSAME...')
            svm_lr_psame = calc_psame(y_pred_svm, y_pred_lr)
            self.log.info('Predicting LR||KNN PSAME...')
            lr_knn_psame = calc_psame(y_pred_lr, y_pred_knn)

            self.log.info('Predicting SVM confidence...')
            svm_confidence        = y_prob_svm.max(axis=1)
            svm_confidence_avg    = np.average(svm_confidence)
            svm_confidence_median = np.median(svm_confidence)
            self.log.info('Predicting LR confidence...')
            lr_confidence         = y_prob_lr.max(axis=1)
            lr_confidence_avg     = np.average(lr_confidence)
            lr_confidence_median  = np.median(lr_confidence)
            self.log.info('Predicting KNN confidence...')
            knn_confidence        = y_prob_knn.max(axis=1)
            knn_confidence_avg     = np.average(knn_confidence)
            knn_confidence_median  = np.median(knn_confidence)

            self.log.info('Calculate KL divergences...')
            np.place(y_prob_svm, y_prob_svm == 0.0, [eps])
            np.place(y_prob_knn, y_prob_knn == 0.0, [eps])
            np.place(y_prob_lr , y_prob_lr  == 0.0, [eps])

            svm_knn_kl_div  = entropy(y_prob_svm  , y_prob_knn)
            svm_knn_kl_div2 = entropy(y_prob_knn  , y_prob_svm)
            svm_knn_kl_div3 = entropy(y_prob_svm.T, y_prob_knn.T)
            svm_knn_kl_div4 = entropy(y_prob_knn.T, y_prob_svm.T)
            svm_knn_kl_div_avg     = np.average(svm_knn_kl_div)
            svm_knn_kl_div2_avg    = np.average(svm_knn_kl_div2)
            svm_knn_kl_div3_avg    = np.average(svm_knn_kl_div3)
            svm_knn_kl_div4_avg    = np.average(svm_knn_kl_div4)
            svm_knn_kl_div3_median = np.median(svm_knn_kl_div3)
            svm_knn_kl_div4_median = np.median(svm_knn_kl_div4)

            svm_lr_kl_div  = entropy(y_prob_svm, y_prob_lr)
            svm_lr_kl_div2 = entropy(y_prob_lr , y_prob_svm)
            svm_lr_kl_div3 = entropy(y_prob_svm.T, y_prob_lr.T)
            svm_lr_kl_div4 = entropy(y_prob_lr.T, y_prob_svm.T)
            svm_lr_kl_div_avg     = np.average(svm_lr_kl_div)
            svm_lr_kl_div2_avg    = np.average(svm_lr_kl_div2)
            svm_lr_kl_div3_avg    = np.average(svm_lr_kl_div3)
            svm_lr_kl_div4_avg    = np.average(svm_lr_kl_div4)
            svm_lr_kl_div3_median = np.median(svm_lr_kl_div3)
            svm_lr_kl_div4_median = np.median(svm_lr_kl_div4)

            lr_knn_kl_div  = entropy(y_prob_lr, y_prob_knn)
            lr_knn_kl_div2 = entropy(y_prob_knn, y_prob_lr)
            lr_knn_kl_div3 = entropy(y_prob_lr.T, y_prob_knn.T)
            lr_knn_kl_div4 = entropy(y_prob_knn.T, y_prob_lr.T)
            lr_knn_kl_div_avg     = np.average(lr_knn_kl_div)
            lr_knn_kl_div2_avg    = np.average(lr_knn_kl_div2)
            lr_knn_kl_div3_avg    = np.average(lr_knn_kl_div3)
            lr_knn_kl_div4_avg    = np.average(lr_knn_kl_div4)
            lr_knn_kl_div3_median = np.median(lr_knn_kl_div3)
            lr_knn_kl_div4_median = np.median(lr_knn_kl_div4)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_score', svm_score, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_score' , lr_score , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/knn_score', knn_score, self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_psame', svm_knn_psame, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_psame' , svm_lr_psame , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_psame' , lr_knn_psame , self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_confidence_avg'   , svm_confidence_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_confidence_median', svm_confidence_median, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_confidence_avg'    , lr_confidence_avg    , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_confidence_median' , lr_confidence_median , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/knn_confidence_avg'   , knn_confidence_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/knn_confidence_median', knn_confidence_median, self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div_avg'    , svm_knn_kl_div_avg    , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div2_avg'   , svm_knn_kl_div2_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div3_avg'   , svm_knn_kl_div3_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div4_avg'   , svm_knn_kl_div4_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div3_median', svm_knn_kl_div3_median, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div4_median', svm_knn_kl_div4_median, self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div_avg'    , svm_lr_kl_div_avg    , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div2_avg'   , svm_lr_kl_div2_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div3_avg'   , svm_lr_kl_div3_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div4_avg'   , svm_lr_kl_div4_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div3_median', svm_lr_kl_div3_median, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div4_median', svm_lr_kl_div4_median, self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div_avg'    , lr_knn_kl_div_avg  , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div2_avg'   , lr_knn_kl_div2_avg , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div3_avg'   , lr_knn_kl_div3_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div4_avg'   , lr_knn_kl_div4_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div3_median', lr_knn_kl_div3_median, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div4_median', lr_knn_kl_div4_median, self.global_step)

            if not self.eval_trainset:
                return

            self.log.info('Predicting train labels from SVM model...')
            y_prob_svm = self.svm.predict_proba(X_train_features)
            y_pred_svm = y_prob_svm.argmax(axis=1)
            svm_score = np.average(y_train == y_pred_svm)
            self.log.info('Predicting train labels from Logistic Regression model...')
            y_prob_lr = self.lr.predict_proba(X_train_features)
            y_pred_lr = y_prob_lr.argmax(axis=1)
            lr_score = np.average(y_train == y_pred_lr)
            self.log.info('Predicting train labels from KNN model...')
            y_prob_knn = self.knn_predict_proba_for_trainset(self.knn_train, X_train_features, y_train)
            y_pred_knn = y_prob_knn.argmax(axis=1)
            knn_score = np.average(y_train == y_pred_knn)

            self.log.info('Predicting SVM||KNN PSAME for trainset...')
            svm_knn_psame = calc_psame(y_pred_svm, y_pred_knn)
            self.log.info('Predicting SVM||LR PSAME for trainset...')
            svm_lr_psame = calc_psame(y_pred_svm, y_pred_lr)
            self.log.info('Predicting LR||KNN PSAME for trainset...')
            lr_knn_psame = calc_psame(y_pred_lr, y_pred_knn)

            self.log.info('Predicting SVM confidence for trainset...')
            svm_confidence        = y_prob_svm.max(axis=1)
            svm_confidence_avg    = np.average(svm_confidence)
            svm_confidence_median = np.median(svm_confidence)
            self.log.info('Predicting LR confidence for trainset...')
            lr_confidence         = y_prob_lr.max(axis=1)
            lr_confidence_avg     = np.average(lr_confidence)
            lr_confidence_median  = np.median(lr_confidence)
            self.log.info('Predicting KNN confidence for trainset...')
            knn_confidence        = y_prob_knn.max(axis=1)
            knn_confidence_avg     = np.average(knn_confidence)
            knn_confidence_median  = np.median(knn_confidence)

            self.log.info('Calculate KL divergences for trainset...')
            np.place(y_prob_svm, y_prob_svm == 0.0, [eps])
            np.place(y_prob_knn, y_prob_knn == 0.0, [eps])
            np.place(y_prob_lr , y_prob_lr  == 0.0, [eps])

            svm_knn_kl_div  = entropy(y_prob_svm  , y_prob_knn)
            svm_knn_kl_div2 = entropy(y_prob_knn  , y_prob_svm)
            svm_knn_kl_div3 = entropy(y_prob_svm.T, y_prob_knn.T)
            svm_knn_kl_div4 = entropy(y_prob_knn.T, y_prob_svm.T)
            svm_knn_kl_div_avg     = np.average(svm_knn_kl_div)
            svm_knn_kl_div2_avg    = np.average(svm_knn_kl_div2)
            svm_knn_kl_div3_avg    = np.average(svm_knn_kl_div3)
            svm_knn_kl_div4_avg    = np.average(svm_knn_kl_div4)
            svm_knn_kl_div3_median = np.median(svm_knn_kl_div3)
            svm_knn_kl_div4_median = np.median(svm_knn_kl_div4)

            svm_lr_kl_div  = entropy(y_prob_svm, y_prob_lr)
            svm_lr_kl_div2 = entropy(y_prob_lr , y_prob_svm)
            svm_lr_kl_div3 = entropy(y_prob_svm.T, y_prob_lr.T)
            svm_lr_kl_div4 = entropy(y_prob_lr.T, y_prob_svm.T)
            svm_lr_kl_div_avg     = np.average(svm_lr_kl_div)
            svm_lr_kl_div2_avg    = np.average(svm_lr_kl_div2)
            svm_lr_kl_div3_avg    = np.average(svm_lr_kl_div3)
            svm_lr_kl_div4_avg    = np.average(svm_lr_kl_div4)
            svm_lr_kl_div3_median = np.median(svm_lr_kl_div3)
            svm_lr_kl_div4_median = np.median(svm_lr_kl_div4)

            lr_knn_kl_div  = entropy(y_prob_lr, y_prob_knn)
            lr_knn_kl_div2 = entropy(y_prob_knn, y_prob_lr)
            lr_knn_kl_div3 = entropy(y_prob_lr.T, y_prob_knn.T)
            lr_knn_kl_div4 = entropy(y_prob_knn.T, y_prob_lr.T)
            lr_knn_kl_div_avg     = np.average(lr_knn_kl_div)
            lr_knn_kl_div2_avg    = np.average(lr_knn_kl_div2)
            lr_knn_kl_div3_avg    = np.average(lr_knn_kl_div3)
            lr_knn_kl_div4_avg    = np.average(lr_knn_kl_div4)
            lr_knn_kl_div3_median = np.median(lr_knn_kl_div3)
            lr_knn_kl_div4_median = np.median(lr_knn_kl_div4)

            suffix = '_trainset'

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_score' + suffix, svm_score, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_score' + suffix , lr_score , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/knn_score' + suffix, knn_score, self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_psame' + suffix, svm_knn_psame, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_psame' + suffix , svm_lr_psame , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_psame' + suffix , lr_knn_psame , self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_confidence_avg' + suffix   , svm_confidence_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_confidence_median' + suffix, svm_confidence_median, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_confidence_avg' + suffix    , lr_confidence_avg    , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_confidence_median' + suffix , lr_confidence_median , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/knn_confidence_avg' + suffix   , knn_confidence_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/knn_confidence_median' + suffix, knn_confidence_median, self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div_avg' + suffix    , svm_knn_kl_div_avg    , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div2_avg' + suffix   , svm_knn_kl_div2_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div3_avg' + suffix   , svm_knn_kl_div3_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div4_avg' + suffix   , svm_knn_kl_div4_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div3_median' + suffix, svm_knn_kl_div3_median, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_knn_kl_div4_median' + suffix, svm_knn_kl_div4_median, self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div_avg' + suffix    , svm_lr_kl_div_avg    , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div2_avg' + suffix   , svm_lr_kl_div2_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div3_avg' + suffix   , svm_lr_kl_div3_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div4_avg' + suffix   , svm_lr_kl_div4_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div3_median' + suffix, svm_lr_kl_div3_median, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/svm_lr_kl_div4_median' + suffix, svm_lr_kl_div4_median, self.global_step)

            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div_avg' + suffix    , lr_knn_kl_div_avg  , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div2_avg' + suffix   , lr_knn_kl_div2_avg , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div3_avg' + suffix   , lr_knn_kl_div3_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div4_avg' + suffix   , lr_knn_kl_div4_avg   , self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div3_median' + suffix, lr_knn_kl_div3_median, self.global_step)
            self.tb_logger_test.log_scalar(self.tested_layer + '/lr_knn_kl_div4_median' + suffix, lr_knn_kl_div4_median, self.global_step)

            self.summary_writer_test.flush()
            return
        elif self.decision_method == 'knn_nc_dropout_sum':
            self.log.info('Predicting test set labels from KNN model using NC dropout...')
            number_of_predictions = 20
            test_knn_predictions_prob_mat = np.zeros(shape=[number_of_predictions, self.dataset.test_set_size, self.num_classes], dtype=np.float32)
            for i in xrange(number_of_predictions):
                self.log.info('Calculating NC dropout - iteration #{}'.format(i+1))
                # collect new features using dropout=0.5
                (X_test_features, ) = \
                    collect_features(
                        agent=self,
                        dataset_name='test',
                        fetches=[self.model.net['embedding_layer']],
                        feed_dict={self.model.dropout_keep_prob: 0.5})
                X_test_features = self.apply_pca(X_test_features, fit=False)
                test_knn_predictions_prob_tmp = self.knn.predict_proba(X_test_features)
                if self.debug_mode:
                    print('test_knn_predictions_prob_tmp[0] for i={}: {}\ny_test[0]={}'.format(i, test_knn_predictions_prob_tmp[0], y_test[0]))
                test_knn_predictions_prob_mat[i] += test_knn_predictions_prob_tmp

            self.log.info("Summing all knn probability vectors")
            test_knn_predictions_prob = np.sum(test_knn_predictions_prob_mat, axis=0)
            y_pred = test_knn_predictions_prob.argmax(axis=1)
        elif self.decision_method == 'dnn_knn_mutual_agreement':
            y_pred = y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
            self.log.info('Predicting test set labels from KNN model...')
            test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
            y_pred_knn = test_knn_predictions_prob.argmax(axis=1)
            ma_score, md_score = calc_mutual_agreement(y_pred_dnn, y_pred_knn, y_test)

            score_str = 'score_metrics/K={}/PCA={}/norm={}/weights={}/decision_method={}'\
                .format(self.knn_neighbors, self.pca_embedding_dims, self.knn_norm, self.knn_weights, self.decision_method)
            self.tb_logger_test.log_scalar(score_str + '/ma_score', ma_score, self.global_step)
            self.tb_logger_test.log_scalar(score_str + '/md_score', md_score, self.global_step)
            print_str = '{}: ma_score={}, md_score={}'.format(score_str, ma_score, md_score)
            self.log.info(print_str)
            print(print_str)
            self.summary_writer_test.flush()
        elif self.decision_method == 'dnn_knn_generalization':
            # training predictions
            train_y_pred_dnn = train_dnn_predictions_prob.argmax(axis=1)
            self.log.info('Predicting train set labels from KNN model...')
            train_knn_predictions_prob = self.knn.predict_proba(X_train_features)
            train_y_pred_knn = train_knn_predictions_prob.argmax(axis=1)
            train_dnn_accuracy = np.sum(train_y_pred_dnn == y_train) / self.dataset.train_set_size
            train_knn_accuracy = np.sum(train_y_pred_knn == y_train) / self.dataset.train_set_size

            # testing predictions
            y_pred = y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
            self.log.info('Predicting test set labels from KNN model...')
            test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
            y_pred_knn = test_knn_predictions_prob.argmax(axis=1)
            dnn_accuracy = np.sum(y_pred_dnn == y_test) / self.dataset.test_set_size
            knn_accuracy = np.sum(y_pred_knn == y_test) / self.dataset.test_set_size

            # calculating generalization
            train_dnn_error_rate = 1.0 - train_dnn_accuracy
            train_knn_error_rate = 1.0 - train_knn_accuracy
            dnn_error_rate       = 1.0 - dnn_accuracy
            knn_error_rate       = 1.0 - knn_accuracy
            dnn_generalization_error = dnn_error_rate - train_dnn_error_rate
            knn_generalization_error = knn_error_rate - train_knn_error_rate

            print('train_dnn_accuracy: {}, train_knn_accuracy: {}\n'.format(train_dnn_accuracy, train_knn_accuracy),
                  'np.sum(train_y_pred_dnn == y_train) = {}\n'.format(np.sum(train_y_pred_dnn == y_train)),
                  'np.sum(train_y_pred_knn == y_train) = {}\n'.format(np.sum(train_y_pred_knn == y_train)),
                  'np.sum(y_pred_dnn == y_test) = {}\n'.format(np.sum(y_pred_dnn == y_test)),
                  'np.sum(y_pred_knn == y_test) = {}\n'.format(np.sum(y_pred_knn == y_test)),
                  'dnn_accuracy: {}, knn_accuracy: {}\n'.format(dnn_accuracy, knn_accuracy),
                  'dnn_error_rate: {}, knn_error_rate: {}\n'.format(dnn_error_rate, knn_error_rate),
                  'DNN generalization: {}, KNN generalization: {}\n'.format(dnn_generalization_error, knn_generalization_error))
            exit(0)

        accuracy = np.sum(y_pred==y_test)/self.dataset.test_set_size

        # writing summaries
        score_str = 'score_metrics/layer={}/decision_method={}/norm={}/PCA={}'\
            .format(self.tested_layer, self.decision_method, self.knn_norm, self.pca_embedding_dims)
        self.tb_logger_test.log_scalar(score_str, accuracy, self.global_step)
        print_str = '{}: accuracy={}.'.format(score_str, accuracy)
        self.log.info(print_str)
        print(print_str)
        self.summary_writer_test.flush()
        self.log.info('Tester {} is done'.format(str(self)))

    def print_stats(self):
        '''print basic test parameters'''
        super(KNNClassifierTester, self).print_stats()
        self.log.info(' DECISION_METHOD: {}'.format(self.decision_method))
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))
        self.log.info(' KNN_NEIGHBORS: {}'.format(self.knn_neighbors))
        self.log.info(' KNN_NORM: {}'.format(self.knn_norm))
        self.log.info(' KNN_WEIGHTS: {}'.format(self.knn_weights))
        self.log.info(' KNN_JOBS: {}'.format(self.knn_jobs))


