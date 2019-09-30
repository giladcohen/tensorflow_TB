from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow_TB.lib.testers.multi_knn_classifier_tester import MultiKNNClassifierTester
from tensorflow_TB.utils.misc import collect_features

eps = 0.000001

class BayesianMultiKNNClassifierTester(MultiKNNClassifierTester):

    def __init__(self, *args, **kwargs):
        super(BayesianMultiKNNClassifierTester, self).__init__(*args, **kwargs)

        self.pre_cstr = 'knn/dropout={}/'.format(self.prm.network.system.DROPOUT_KEEP_PROB)

        self.knn_accumulated_pred_proba = {}
        self.pred_proba                 = {}
        for k in self.k_list:
            self.knn_accumulated_pred_proba[k] = np.zeros((self.dataset.test_set_size, self.dataset.num_classes))

    def test(self):
        X_train_features, \
        X_test_features, \
        train_dnn_predictions_prob, \
        test_dnn_predictions_prob, \
        y_train, \
        y_test = self.fetch_dump_data_features()

        X_train_features = self.apply_pca(X_train_features, fit=True)
        X_test_features = self.apply_pca(X_test_features, fit=False)

        # fitting knn models
        for k in self.k_list:
            self.log.info('Fitting KNN model for k={}...'.format(k))
            self.knn_dict[k].fit(X_train_features, y_train)

        self.log.info('Predicting test set labels from DNN model...')
        y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
        dnn_score = np.average(y_test == y_pred_dnn)
        self.log.info('Calculate DNN test confidence scores...')
        confidence        = test_dnn_predictions_prob.max(axis=1)
        confidence_avg    = np.average(confidence)
        confidence_median = np.median(confidence)

        np.place(test_dnn_predictions_prob, test_dnn_predictions_prob == 0.0, [eps])  # for KL divergences
        self.tb_logger_test.log_scalar('dnn_score'            , dnn_score        , self.global_step)
        self.tb_logger_test.log_scalar('dnn_confidence_avg'   , confidence_avg   , self.global_step)
        self.tb_logger_test.log_scalar('dnn_confidence_median', confidence_median, self.global_step)

        # now iterating over the knn models
        self.log.info('Running Bayesian network for features with DROPOUT_KEEP_PROB={}\n'.
                      format(self.prm.network.system.DROPOUT_KEEP_PROB))
        total_iterations_cnt = 0
        while total_iterations_cnt < 20:
            (tmp_features,) = collect_features(
                agent=self,
                dataset_name='test',
                fetches=[self.model.net[self.tested_layer]],
                feed_dict={self.model.dropout_keep_prob: self.prm.network.system.DROPOUT_KEEP_PROB})
            tmp_features = self.apply_pca(tmp_features, fit=False)
            for k in self.k_list:
                tmp_pred_proba = self.knn_dict[k].predict_proba(tmp_features)
                self.knn_accumulated_pred_proba[k] += tmp_pred_proba
            total_iterations_cnt += 1

        for k in self.k_list:
            self.pred_proba[k] = self.knn_accumulated_pred_proba[k] / total_iterations_cnt

        # dumping
        for k in self.k_list:
            self.process(
                model=self.knn_dict[k],
                dataset_name='test',
                X=self.pred_proba[k],
                y=y_test,
                dnn_predictions_prob=test_dnn_predictions_prob)

        self.summary_writer_test.flush()

    def predict_proba(self, X, model):
        return X
