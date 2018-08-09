from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.classification_trainer import ClassificationTrainer
import numpy as np
from utils.misc import collect_features, calc_mutual_agreement, calc_psame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
# from skl_groups.divergences import KNNDivergenceEstimator
from scipy.stats import entropy

eps = 0.0001

class ClassificationMetricsTrainer(ClassificationTrainer):
    """Implementing classification trainer with many different metrics"""

    def __init__(self, *args, **kwargs):
        super(ClassificationMetricsTrainer, self).__init__(*args, **kwargs)
        self.pca_reduction         = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims    = self.prm.train.train_control.PCA_EMBEDDING_DIMS

        self.randomized_dataset = False
        self.eval_trainset      = False
        self.collect_knn        = True
        self.collect_svm        = False
        self.collect_lr         = False

        if self.randomized_dataset:
            self.train_handle = 'train_random'
            self.train_eval_handle = 'train_random_eval'
        else:
            self.train_handle = 'train'
            self.train_eval_handle = 'train_eval'

        # KNN testing parameters
        self.knn_neighbors   = self.prm.test.test_control.KNN_NEIGHBORS
        self.knn_norm        = self.prm.test.test_control.KNN_NORM
        self.knn_weights     = self.prm.test.test_control.KNN_WEIGHTS
        self.knn_jobs        = self.prm.test.test_control.KNN_JOBS

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

        self.svm = LinearSVC(
            penalty=self.knn_norm.lower(),
            dual=False,
            random_state=self.rand_gen)

        self.lr = LogisticRegression(
            penalty=self.knn_norm.lower(),
            dual=False,
            random_state=self.rand_gen,
            n_jobs=self.knn_jobs)

        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)

    def train_step(self):
        '''Implementing one training step'''
        _, images, labels = self.dataset.get_mini_batch(self.train_handle, self.plain_sess)
        _, self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
                                            feed_dict={self.model.images: images,
                                                       self.model.labels: labels,
                                                       self.model.is_training: True})

    def apply_pca(self, X, fit=False):
        """If pca_reduction is True, apply PCA reduction"""
        if self.pca_reduction:
            self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.model.embedding_dims, self.pca_embedding_dims))
            if fit:
                self.pca.fit(X)
            X = self.pca.transform(X)
        return X

    def knn_predict_proba_for_trainset(self, X_train_features, y_train):
        """
        :param X_train_features: Training set features ([n_samples, n_features])
        :param y_train: training set gt ([n_samples])
        :return: knn predictions of the training set, using an efficient leave-one-out.
        """
        biased_knn_predictions_prob_train = self.knn_train.predict_proba(X_train_features)
        knn_predictions_prob_train = np.zeros(y_train.shape)

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

    def test_step(self):
        '''Implementing one test step.'''
        self.log.info('start running test within training. global_step={}'.format(self.global_step))
        self.log.info('Collecting {} {} set embedding features'.format(self.dataset.train_set_size, self.train_eval_handle))
        (X_train_features, y_train, train_dnn_predictions_prob) = \
            collect_features(
                agent=self,
                dataset_name=self.train_eval_handle,
                fetches=[self.model.net['embedding_layer'], self.model.labels, self.model.predictions_prob],
                feed_dict={self.model.dropout_keep_prob: 1.0})

        self.log.info('Collecting {} test set embedding features and DNN predictions'.format(self.dataset.test_set_size))
        (X_test_features, y_test, test_dnn_predictions_prob) = \
            collect_features(
                agent=self,
                dataset_name='test',
                fetches=[self.model.net['embedding_layer'], self.model.labels, self.model.predictions_prob],
                feed_dict={self.model.dropout_keep_prob: 1.0})

        X_train_features = self.apply_pca(X_train_features, fit=True)
        X_test_features  = self.apply_pca(X_test_features , fit=False)

        # fittings
        if self.collect_knn:
            self.log.info('Fitting KNN model...')
            self.knn.fit(X_train_features, y_train)
        if self.collect_svm:
            self.log.info('Fitting SVM model...')
            self.svm.fit(X_train_features, y_train)
        if self.collect_lr:
            self.log.info('Fitting Logistic Regression model...')
            self.lr.fit(X_train_features, y_train)

        # predictions (test set)
        self.log.info('Predicting test set labels from DNN model...')
        y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
        if self.collect_knn:
            self.log.info('Predicting test set labels from KNN model...')
            test_knn_predictions_prob = self.knn.predict_proba(X_test_features)
            y_pred_knn = test_knn_predictions_prob.argmax(axis=1)
        if self.collect_svm:
            self.log.info('Predicting test set labels from SVM model...')
            y_pred_svm = self.svm.predict(X_test_features)
        if self.collect_lr:
            self.log.info('Predicting test set labels from Logistic Regression model...')
            test_lr_predictions_prob = self.lr.predict_proba(X_test_features)
            y_pred_lr = test_lr_predictions_prob.argmax(axis=1)

        # calculate metrics
        self.log.info('Calculate test set scores...')
        dnn_score = np.average(y_test == y_pred_dnn)
        if self.collect_knn:
            knn_score = np.average(y_test == y_pred_knn)
        if self.collect_svm:
            svm_score = np.average(y_test == y_pred_svm)
        if self.collect_lr:
            lr_score  = np.average(y_test == y_pred_lr)

        self.log.info('Calculate ma/md and psame scores...')
        if self.collect_knn:
            ma_score_knn, md_score_knn = calc_mutual_agreement(y_pred_dnn, y_pred_knn, y_test)
        if self.collect_svm:
            ma_score_svm, md_score_svm = calc_mutual_agreement(y_pred_dnn, y_pred_svm, y_test)
        if self.collect_lr:
            ma_score_lr , md_score_lr  = calc_mutual_agreement(y_pred_dnn, y_pred_lr , y_test)
        if self.collect_knn:
            psame_knn = calc_psame(y_pred_dnn, y_pred_knn)
        if self.collect_svm:
            psame_svm = calc_psame(y_pred_dnn, y_pred_svm)
        if self.collect_lr:
            psame_lr  = calc_psame(y_pred_dnn, y_pred_lr)

        self.log.info('Calculate KL divergences...')
        np.place(test_dnn_predictions_prob, test_dnn_predictions_prob == 0.0, [eps])
        if self.collect_knn:
            np.place(test_knn_predictions_prob, test_knn_predictions_prob == 0.0, [eps])
            dnn_knn_kl_div = entropy(test_dnn_predictions_prob, test_knn_predictions_prob)
            dnn_knn_kl_div_avg = np.average(dnn_knn_kl_div)
        if self.collect_lr:
            np.place(test_lr_predictions_prob, test_lr_predictions_prob == 0.0, [eps])
            dnn_lr_kl_div  = entropy(test_dnn_predictions_prob, test_lr_predictions_prob)
            dnn_lr_kl_div_avg  = np.average(dnn_lr_kl_div)

        if self.eval_trainset:
            # special fitting
            if self.collect_knn:
                self.log.info('Fitting KNN model...')
                self.knn_train.fit(X_train_features, y_train)

            # predictions (train set)
            self.log.info('Predicting train set labels from DNN model...')
            y_pred_dnn_train = train_dnn_predictions_prob.argmax(axis=1)
            if self.collect_knn:
                self.log.info('Predicting train set labels from KNN model...')
                train_knn_predictions_prob = self.knn_predict_proba_for_trainset(X_train_features, y_train)
                y_pred_knn_train = train_knn_predictions_prob.argmax(axis=1)
            if self.collect_svm:
                self.log.info('Predicting train set labels from SVM model...')
                y_pred_svm_train = self.svm.predict(X_train_features)
            if self.collect_lr:
                self.log.info('Predicting train set labels from Logistic Regression model...')
                train_lr_predictions_prob = self.lr.predict_proba(X_train_features)
                y_pred_lr_train = train_lr_predictions_prob.argmax(axis=1)

            # calculate metrics
            self.log.info('Calculate train set scores...')
            dnn_score_train = np.average(y_train == y_pred_dnn_train)
            if self.collect_knn:
                knn_score_train = np.average(y_train == y_pred_knn_train)
            if self.collect_svm:
                svm_score_train = np.average(y_train == y_pred_svm_train)
            if self.collect_lr:
                lr_score_train  = np.average(y_train == y_pred_lr_train)
            self.log.info('Calculate ma/md and psame scores...')
            if self.collect_knn:
                ma_score_knn_train, md_score_knn_train = calc_mutual_agreement(y_pred_dnn_train, y_pred_knn_train, y_train)
            if self.collect_svm:
                ma_score_svm_train, md_score_svm_train = calc_mutual_agreement(y_pred_dnn_train, y_pred_svm_train, y_train)
            if self.collect_lr:
                ma_score_lr_train , md_score_lr_train  = calc_mutual_agreement(y_pred_dnn_train, y_pred_lr_train , y_train)
            if self.collect_knn:
                psame_knn_train = calc_psame(y_pred_dnn_train, y_pred_knn_train)
            if self.collect_svm:
                psame_svm_train = calc_psame(y_pred_dnn_train, y_pred_svm_train)
            if self.collect_lr:
                psame_lr_train  = calc_psame(y_pred_dnn_train, y_pred_lr_train)
            self.log.info('Calculate KL divergences...')
            np.place(train_dnn_predictions_prob, train_dnn_predictions_prob == 0.0, [eps])
            if self.collect_knn:
                np.place(train_knn_predictions_prob, train_knn_predictions_prob == 0.0, [eps])
                dnn_knn_kl_div_train = entropy(train_dnn_predictions_prob, train_knn_predictions_prob)
                dnn_knn_kl_div_avg_train = np.average(dnn_knn_kl_div_train)
            if self.collect_lr:
                np.place(train_lr_predictions_prob, train_lr_predictions_prob == 0.0, [eps])
                dnn_lr_kl_div_train  = entropy(train_dnn_predictions_prob, train_lr_predictions_prob)
                dnn_lr_kl_div_avg_train = np.average(dnn_lr_kl_div_train)

        self.test_retention.add_score(dnn_score, self.global_step)

        # save summaries
        self.tb_logger_test.log_scalar('dnn_score', dnn_score, self.global_step)
        if self.collect_knn:
            self.tb_logger_test.log_scalar('knn_score', knn_score, self.global_step)
        if self.collect_svm:
            self.tb_logger_test.log_scalar('svm_score', svm_score, self.global_step)
        if self.collect_lr:
            self.tb_logger_test.log_scalar('lr_score' , lr_score , self.global_step)

        if self.collect_knn:
            self.tb_logger_test.log_scalar('knn_ma_score', ma_score_knn, self.global_step)
            self.tb_logger_test.log_scalar('knn_md_score', md_score_knn, self.global_step)
        if self.collect_svm:
            self.tb_logger_test.log_scalar('svm_ma_score', ma_score_svm, self.global_step)
            self.tb_logger_test.log_scalar('svm_md_score', md_score_svm, self.global_step)
        if self.collect_lr:
            self.tb_logger_test.log_scalar('lr_ma_score' , ma_score_lr , self.global_step)
            self.tb_logger_test.log_scalar('lr_md_score' , md_score_lr , self.global_step)
        if self.collect_knn:
            self.tb_logger_test.log_scalar('knn_psame'   , psame_knn   , self.global_step)
        if self.collect_svm:
            self.tb_logger_test.log_scalar('svm_psame'   , psame_svm   , self.global_step)
        if self.collect_lr:
            self.tb_logger_test.log_scalar('lr_psame'    , psame_lr   , self.global_step)

        if self.collect_knn:
            self.tb_logger_test.log_scalar('dnn_knn_kl_div_avg', dnn_knn_kl_div_avg, self.global_step)
        if self.collect_lr:
            self.tb_logger_test.log_scalar('dnn_lr_kl_div_avg' , dnn_lr_kl_div_avg , self.global_step)

        if self.eval_trainset:
            self.tb_logger_test.log_scalar('dnn_score_trainset', dnn_score_train, self.global_step)
            if self.collect_knn:
                self.tb_logger_test.log_scalar('knn_score_trainset', knn_score_train, self.global_step)
            if self.collect_svm:
                self.tb_logger_test.log_scalar('svm_score_trainset', svm_score_train, self.global_step)
            if self.collect_lr:
                self.tb_logger_test.log_scalar('lr_score_trainset' , lr_score_train, self.global_step)

            if self.collect_knn:
                self.tb_logger_test.log_scalar('knn_ma_score_trainset', ma_score_knn_train, self.global_step)
                self.tb_logger_test.log_scalar('knn_md_score_trainset', md_score_knn_train, self.global_step)
            if self.collect_svm:
                self.tb_logger_test.log_scalar('svm_ma_score_trainset', ma_score_svm_train, self.global_step)
                self.tb_logger_test.log_scalar('svm_md_score_trainset', md_score_svm_train, self.global_step)
            if self.collect_lr:
                self.tb_logger_test.log_scalar('lr_ma_score_trainset' , ma_score_lr_train, self.global_step)
                self.tb_logger_test.log_scalar('lr_md_score_trainset' , md_score_lr_train, self.global_step)
            if self.collect_knn:
                self.tb_logger_test.log_scalar('knn_psame_trainset', psame_knn_train, self.global_step)
            if self.collect_svm:
                self.tb_logger_test.log_scalar('svm_psame_trainset', psame_svm_train, self.global_step)
            if self.collect_lr:
                self.tb_logger_test.log_scalar('lr_psame_trainset' , psame_lr_train, self.global_step)

            if self.collect_knn:
                self.tb_logger_test.log_scalar('dnn_knn_kl_div_avg_trainset', dnn_knn_kl_div_avg_train, self.global_step)
            if self.collect_lr:
                self.tb_logger_test.log_scalar('dnn_lr_kl_div_avg_trainset' , dnn_lr_kl_div_avg_train , self.global_step)

        self.summary_writer_test.flush()

    def to_test(self):
        ret = self.global_step % self.test_steps == 0
        # ret = ret or (self.global_step < 100 and self.global_step % 10 == 0)
        ret = ret and self._activate_test
        return ret


