from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.testers.multi_knn_classifier_tester import MultiKNNClassifierTester
from utils.misc import collect_features

eps = 0.000001

class BayesianMultiKNNClassifierTester(MultiKNNClassifierTester):

    def __init__(self, *args, **kwargs):
        super(BayesianMultiKNNClassifierTester, self).__init__(*args, **kwargs)

        self.pre_cstr = 'knn/dropout={}/'.format(self.prm.network.system.DROPOUT_KEEP_PROB)

    def predict_proba(self, X, model):
        if model == 'dnn':
            return X  # do nothing, bypassing the probabilities
        else:
            # we don't care about X, collecting the (Bayesian) test features from scratch
            self.log.info('Running Bayesian network for features with DROPOUT_KEEP_PROB={} for model {}\n'.
                          format(self.prm.network.system.DROPOUT_KEEP_PROB, str(model)))
            accumulated_pred_proba = np.zeros((self.dataset.test_set_size, self.dataset.num_classes))
            total_neighbors_cnt  = 0
            total_iterations_cnt = 0
            while total_neighbors_cnt < 100000:
                (tmp_features, ) = collect_features(
                    agent=self,
                    dataset_name='test',
                    fetches=[self.model.net[self.tested_layer]],
                    feed_dict={self.model.dropout_keep_prob: self.prm.network.system.DROPOUT_KEEP_PROB})
                tmp_features = self.apply_pca(tmp_features, fit=False)
                tmp_pred_proba = model.predict_proba(tmp_features)
                accumulated_pred_proba += tmp_pred_proba
                total_neighbors_cnt  += model.n_neighbors
                total_iterations_cnt += 1
            pred_proba = accumulated_pred_proba / total_iterations_cnt
            return pred_proba
