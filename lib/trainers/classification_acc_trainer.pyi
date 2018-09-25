# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# from lib.trainers.classification_trainer import ClassificationTrainer
# import numpy as np
# from utils.misc import collect_features, calc_mutual_agreement, calc_psame
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.decomposition import PCA
# # from skl_groups.divergences import KNNDivergenceEstimator
# from scipy.stats import entropy
#
# eps = 0.000001
#
# class ClassificationMetricsTrainer(ClassificationTrainer):
#     """Implementing classification trainer with many different metrics"""
#
#     def __init__(self, *args, **kwargs):
#         super(ClassificationMetricsTrainer, self).__init__(*args, **kwargs)
#         self.pca_reduction         = self.prm.train.train_control.PCA_REDUCTION
#         self.pca_embedding_dims    = self.prm.train.train_control.PCA_EMBEDDING_DIMS
#
#         self.eval_trainset      = self.prm.test.test_control.EVAL_TRAINSET
#         self.randomized_dataset = 'random' in str(self.dataset)
#         self.collect_knn        = True
#         self.collect_svm        = True
#         self.collect_lr         = True
#
#         if self.randomized_dataset:
#             self.train_handle = 'train_random'
#             self.train_eval_handle = 'train_random_eval'
#         else:
#             self.train_handle = 'train'
#             self.train_eval_handle = 'train_eval'
#
#         # KNN testing parameters
#         self.knn_neighbors   = self.prm.test.test_control.KNN_NEIGHBORS
#         self.knn_norm        = self.prm.test.test_control.KNN_NORM
#         self.knn_weights     = self.prm.test.test_control.KNN_WEIGHTS
#         self.knn_jobs        = self.prm.test.test_control.KNN_JOBS
#
#         if self.knn_norm not in ['L1', 'L2']:
#             err_str = 'knn_norm {} is not supported'.format(self.knn_norm)
#             self.log.error(err_str)
#             raise AssertionError(err_str)
#
#         self.knn = KNeighborsClassifier(
#             n_neighbors=self.knn_neighbors,
#             weights=self.knn_weights,
#             p=int(self.knn_norm[-1]),
#             n_jobs=self.knn_jobs)
#
#         self.knn_train = KNeighborsClassifier(
#             n_neighbors=self.knn_neighbors + 1,
#             weights=self.knn_weights,
#             p=int(self.knn_norm[-1]),
#             n_jobs=self.knn_jobs)
#
#         self.svm = SVC(
#             kernel='linear',
#             probability=True,
#             random_state=self.rand_gen)
#
#         self.lr = LogisticRegression(
#             penalty=self.knn_norm.lower(),
#             dual=False,
#             random_state=self.rand_gen,
#             n_jobs=self.knn_jobs)
#
#         self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)
#
#     def train_step(self):
#         '''Implementing one training step'''
#         _, images, labels = self.dataset.get_mini_batch(self.train_handle, self.plain_sess)
#         _, self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
#                                             feed_dict={self.model.images: images,
#                                                        self.model.labels: labels,
#                                                        self.model.is_training: True})
#
#     def apply_pca(self, X, fit=False):
#         """If pca_reduction is True, apply PCA reduction"""
#         if self.pca_reduction:
#             self.log.info('Reducing features_vec from {} dims to {} dims using PCA'.format(self.model.embedding_dims, self.pca_embedding_dims))
#             if fit:
#                 self.pca.fit(X)
#             X = self.pca.transform(X)
#         return X
#
#     def knn_predict_proba_for_trainset(self, model, X_train_features, y_train):
#         """
#         :param model: knn_train
#         :param X_train_features: Training set features ([n_samples, n_features])
#         :param y_train: training set gt ([n_samples])
#         :return: knn predictions of the training set, using an efficient leave-one-out.
#         """
#         biased_knn_predictions_prob_train = model.predict_proba(X_train_features)
#         knn_predictions_prob_train = np.zeros(biased_knn_predictions_prob_train.shape)
#
#         for i in range(len(X_train_features)):
#             y = int(y_train[i])
#             proba = biased_knn_predictions_prob_train[i]
#             # assert proba[y] >= 1/(self.knn_neighbors + 1), "for i={}: prob[y={}] = {}, but cannot be smaller than {}"\
#             #     .format(i, y, proba[y], 1/(self.knn_neighbors + 1))
#             if proba[y] >= 1/(self.knn_neighbors + 1):
#                 proba[y] -= 1/(self.knn_neighbors + 1)
#                 proba *= (self.knn_neighbors + 1)/self.knn_neighbors
#             else:
#                 self.log.warn("for i={}: prob[y={}] = {}, but cannot be smaller than {}. Just warning."\
#                     .format(i, y, proba[y], 1/(self.knn_neighbors + 1)))
#             assert np.isclose(sum(proba), 1.0), "sum of proba[i={}] is {} instead of 1.0".format(i, sum(proba))
#             knn_predictions_prob_train[i] = proba
#
#         return knn_predictions_prob_train
#
#     def process(self, model_name, dataset_name, X, y, dnn_predictions_prob):
#         """
#         :param model_name: A fitted model name to predict and save metrics for
#         :param dataset_name: 'test' or 'train'
#         :param X: dataset, features.
#         :param y: labels
#         :param dnn_predictions_prob: dnn predictions on the dataset
#         :return: None. Saves metrics.
#         """
#
#         if model_name is 'knn':
#             if dataset_name is 'test':
#                 model = self.knn
#             else:
#                 model = self.knn_train
#         elif model_name is 'svm':
#             model = self.svm
#         elif model_name is 'lr':
#             model = self.lr
#         else:
#             err_str = 'unknown model_name: {}'.format(model_name)
#             self.log.error(err_str)
#             raise AssertionError(err_str)
#
#         y_pred_dnn = dnn_predictions_prob.argmax(axis=1)
#
#         self.log.info('Predicting {} labels for dataset {} using model\n {}...'.format(y.shape[0], dataset_name, str(model)))
#         if model_name is 'knn' and dataset_name is 'train':
#             predictions_prob = self.knn_predict_proba_for_trainset(model, X, y)
#         else:
#             predictions_prob = model.predict_proba(X)
#         y_pred = predictions_prob.argmax(axis=1)
#
#         # calculate metrics
#         self.log.info('Calculate {} set scores for model_name {}...'.format(dataset_name, model_name))
#         score = np.average(y == y_pred)
#
#         self.log.info('Calculate ma/md and psame scores...')
#         ma_score, md_score = calc_mutual_agreement(y_pred_dnn, y_pred, y)
#         psame = calc_psame(y_pred_dnn, y_pred)
#
#         self.log.info('Calculate confidence scores...')
#         confidence = predictions_prob.max(axis=1)
#         confidence_avg    = np.average(confidence)
#         confidence_median = np.median(confidence)
#
#         self.log.info('Calculate KL divergences...')
#         np.place(predictions_prob, predictions_prob == 0.0, [eps])
#         kl_div  = entropy(dnn_predictions_prob, predictions_prob)
#         kl_div2 = entropy(predictions_prob, dnn_predictions_prob)
#         kl_div_avg  = np.average(kl_div)
#         kl_div2_avg = np.average(kl_div2)
#
#         if dataset_name is 'test':
#             suffix = ''
#         else:
#             suffix = '_trainset'
#         self.tb_logger_test.log_scalar(model_name + '_score'            + suffix, score            , self.global_step)
#         self.tb_logger_test.log_scalar(model_name + '_ma_score'         + suffix, ma_score         , self.global_step)
#         self.tb_logger_test.log_scalar(model_name + '_md_score'         + suffix, md_score         , self.global_step)
#         self.tb_logger_test.log_scalar(model_name + '_psame'            + suffix, psame            , self.global_step)
#         self.tb_logger_test.log_scalar(model_name + '_confidence_avg'   + suffix, confidence_avg   , self.global_step)
#         self.tb_logger_test.log_scalar(model_name + '_confidence_median'+ suffix, confidence_median, self.global_step)
#         self.tb_logger_test.log_scalar(model_name + '_kl_div_avg'       + suffix, kl_div_avg       , self.global_step)
#         self.tb_logger_test.log_scalar(model_name + '_kl_div2_avg'      + suffix, kl_div2_avg      , self.global_step)
#
#     def test_step(self):
#         '''Implementing one test step.'''
#         self.log.info('start running test within training. global_step={}'.format(self.global_step))
#         self.log.info('Collecting {} {} set embedding features'.format(self.dataset.train_set_size, self.train_eval_handle))
#         (X_train_features, y_train, train_dnn_predictions_prob) = \
#             collect_features(
#                 agent=self,
#                 dataset_name=self.train_eval_handle,
#                 fetches=[self.model.net['embedding_layer'], self.model.labels, self.model.predictions_prob],
#                 feed_dict={self.model.dropout_keep_prob: 1.0})
#
#         self.log.info('Collecting {} test set embedding features and DNN predictions'.format(self.dataset.test_set_size))
#         (X_test_features, y_test, test_dnn_predictions_prob) = \
#             collect_features(
#                 agent=self,
#                 dataset_name='test',
#                 fetches=[self.model.net['embedding_layer'], self.model.labels, self.model.predictions_prob],
#                 feed_dict={self.model.dropout_keep_prob: 1.0})
#
#         X_train_features = self.apply_pca(X_train_features, fit=True)
#         X_test_features  = self.apply_pca(X_test_features , fit=False)
#
#         # fittings
#         # if self.collect_knn:
#         #     self.log.info('Fitting KNN model...')
#         #     self.knn.fit(X_train_features, y_train)
#         # if self.collect_svm:
#         #     self.log.info('Fitting SVM model...')
#         #     self.svm.fit(X_train_features, y_train)
#         # if self.collect_lr:
#         #     self.log.info('Fitting Logistic Regression model...')
#         #     self.lr.fit(X_train_features, y_train)
#
#         self.log.info('Predicting test set labels from DNN model...')
#         y_pred_dnn = test_dnn_predictions_prob.argmax(axis=1)
#         dnn_score = np.average(y_test == y_pred_dnn)
#         # self.log.info('Calculate DNN test confidence scores...')
#         # confidence        = test_dnn_predictions_prob.max(axis=1)
#         # confidence_avg    = np.average(confidence)
#         # confidence_median = np.median(confidence)
#
#         # np.place(test_dnn_predictions_prob, test_dnn_predictions_prob  == 0.0, [eps])  # for KL divergences
#         self.tb_logger_test.log_scalar('dnn_score'            , dnn_score        , self.global_step)
#         # self.tb_logger_test.log_scalar('dnn_confidence_avg'   , confidence_avg   , self.global_step)
#         # self.tb_logger_test.log_scalar('dnn_confidence_median', confidence_median, self.global_step)
#
#         # for model_name in ['knn', 'svm', 'lr']:
#         #     if not ((model_name is 'knn' and self.collect_knn) or
#         #             (model_name is 'svm' and self.collect_svm) or
#         #             (model_name is 'lr'  and self.collect_lr)):
#         #         continue
#         #     self.process(model_name=model_name,
#         #                  dataset_name='test',
#         #                  X=X_test_features,
#         #                  y=y_test,
#         #                  dnn_predictions_prob=test_dnn_predictions_prob)
#
#         if self.eval_trainset:
#             # if self.collect_knn:
#             #     self.log.info('Fitting KNN model for training set...')
#             #     self.knn_train.fit(X_train_features, y_train)
#
#             self.log.info('Predicting train set labels from DNN model...')
#             y_pred_dnn = train_dnn_predictions_prob.argmax(axis=1)
#             dnn_score = np.average(y_train == y_pred_dnn)
#             # self.log.info('Calculate DNN train confidence scores...')
#             # confidence        = train_dnn_predictions_prob.max(axis=1)
#             # confidence_avg    = np.average(confidence)
#             # confidence_median = np.median(confidence)
#
#             # np.place(train_dnn_predictions_prob, train_dnn_predictions_prob == 0.0, [eps])  # for KL divergences
#             self.tb_logger_test.log_scalar('dnn_score_trainset'            , dnn_score        , self.global_step)
#             # self.tb_logger_test.log_scalar('dnn_confidence_avg_trainset'   , confidence_avg   , self.global_step)
#             # self.tb_logger_test.log_scalar('dnn_confidence_median_trainset', confidence_median, self.global_step)
#
#             # for model_name in ['knn', 'svm', 'lr']:
#             #     if not ((model_name is 'knn' and self.collect_knn) or
#             #             (model_name is 'svm' and self.collect_svm) or
#             #             (model_name is 'lr'  and self.collect_lr)):
#             #         continue
#             #     self.process(model_name=model_name,
#             #                  dataset_name='train',
#             #                  X=X_train_features,
#             #                  y=y_train,
#             #                  dnn_predictions_prob=train_dnn_predictions_prob)
#
#         self.test_retention.add_score(dnn_score, self.global_step)
#         self.summary_writer_test.flush()
#
#     def to_test(self):
#         ret = self.global_step % self.test_steps == 0
#         # ret = ret or (self.global_step < 100 and self.global_step % 10 == 0)
#         ret = ret and self._activate_test
#         return ret
#
#
