from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.testers.knn_classifier_tester import KNNClassifierTester
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import entropy
from utils.misc import calc_psame

eps = 0.000001

class MultiKNNClassifierTester(KNNClassifierTester):

    def __init__(self, *args, **kwargs):
        super(MultiKNNClassifierTester, self).__init__(*args, **kwargs)

        self.k_list = [1, 3, 4, 5, 6, 7, 8, 9, 10,
                       12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                       45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                       110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                       220, 240, 260, 280, 300,
                       350, 400, 450, 500,
                       600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

        # taking maximum of <num_of_training_samples>/<num_of_classes>
        num_of_samples_in_a_class = int(self.prm.dataset.TRAIN_SET_SIZE / self.prm.network.NUM_CLASSES)
        self.k_list = [k for k in self.k_list if k <= num_of_samples_in_a_class]

        # constructing the knn classifiers:
        self.knn_dict = {}
        for k in self.k_list:
            self.knn_dict[k] = KNeighborsClassifier(
                n_neighbors=k,
                weights=self.knn_weights,
                p=int(self.knn_norm[-1]),
                n_jobs=self.knn_jobs
            )

        self.pre_cstr = 'knn/'

    def predict_proba(self, X, model):
        """
        Predict probability of labels for X using the model
        :param X: input, np.ndarray
        :param model: scikit-learn model
        :return: labels probability, np.ndarray
        """
        return model.predict_proba(X)

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
        for k in self.k_list:
            self.process(
                model=self.knn_dict[k],
                dataset_name='test',
                X=X_test_features,
                y=y_test,
                dnn_predictions_prob=test_dnn_predictions_prob)

        self.summary_writer_test.flush()

    def process(self, model, dataset_name, X, y, dnn_predictions_prob):
        """
        :param model: A fitted model name to predict and save metrics for
        :param dataset_name: 'test' or 'train'
        :param X: dataset, features.
        :param y: labels
        :param dnn_predictions_prob: dnn predictions on the dataset
        :return: None. Saves metrics.
        """
        y_pred_dnn = dnn_predictions_prob.argmax(axis=1)

        self.log.info('Predicting {} labels for dataset {} using knn model with k={} and norm=L{}\n {}...'
                      .format(y.shape[0], dataset_name, model.n_neighbors, model.p, str(model)))
        predictions_prob = self.predict_proba(X, model)
        y_pred = predictions_prob.argmax(axis=1)

        # calculate metrics
        self.log.info('Calculate {} set scores...'.format(dataset_name))
        score = np.average(y == y_pred)

        self.log.info('Calculate psame scores...')
        psame = calc_psame(y_pred_dnn, y_pred)

        self.log.info('Calculate confidence scores...')
        confidence = predictions_prob.max(axis=1)
        confidence_avg    = np.average(confidence)
        confidence_median = np.median(confidence)

        self.log.info('Calculate KL divergences...')
        np.place(predictions_prob, predictions_prob == 0.0, [eps])
        kl_div  = entropy(dnn_predictions_prob  , predictions_prob)
        kl_div2 = entropy(predictions_prob      , dnn_predictions_prob)
        kl_div3 = entropy(dnn_predictions_prob.T, predictions_prob.T)
        kl_div4 = entropy(predictions_prob.T    , dnn_predictions_prob.T)

        kl_div_avg     = np.average(kl_div)
        kl_div2_avg    = np.average(kl_div2)
        kl_div3_avg    = np.average(kl_div3)
        kl_div4_avg    = np.average(kl_div4)
        kl_div_median  = np.median(kl_div)
        kl_div2_median = np.median(kl_div2)
        kl_div3_median = np.median(kl_div3)
        kl_div4_median = np.median(kl_div4)

        if dataset_name is 'test':
            suffix = ''
        else:
            suffix = '_trainset'
        cstr = self.pre_cstr + 'k={}/norm=L{}/'.format(model.n_neighbors, model.p)

        self.tb_logger_test.log_scalar(cstr + 'knn_score'             + suffix, score            , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_psame'             + suffix, psame            , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_confidence_avg'    + suffix, confidence_avg   , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_confidence_median' + suffix, confidence_median, self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_kl_div_avg'        + suffix, kl_div_avg       , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_kl_div2_avg'       + suffix, kl_div2_avg      , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_kl_div3_avg'       + suffix, kl_div3_avg      , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_kl_div4_avg'       + suffix, kl_div4_avg      , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_kl_div_median'     + suffix, kl_div_median    , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_kl_div2_median'    + suffix, kl_div2_median   , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_kl_div3_median'    + suffix, kl_div3_median   , self.global_step)
        self.tb_logger_test.log_scalar(cstr + 'knn_kl_div4_median'    + suffix, kl_div4_median   , self.global_step)
