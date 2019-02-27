from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.testers.knn_classifier_tester import KNNClassifierTester
from sklearn.neighbors import NearestNeighbors
from lib.datasets.influence_feeder import Feeder
import matplotlib.pyplot as plt

class KNNInfluenceTester(KNNClassifierTester):

    def __init__(self, *args, **kwargs):
        super(KNNInfluenceTester, self).__init__(*args, **kwargs)

        self.knn = NearestNeighbors(n_neighbors=self.knn_neighbors,  # should be the number of the training set
                                    # algorithm='brute',
                                    p=int(self.knn_norm[-1]),
                                    n_jobs=self.knn_jobs)

        self.feeder = Feeder('influence_feeder', self.prm)

    def test(self):
        # building the feeder
        self.feeder.build()

        # get all train and test embeddings
        X_train_features, \
        X_test_features, \
        train_dnn_predictions_prob, \
        test_dnn_predictions_prob, \
        y_train, \
        y_test = self.fetch_dump_data_features()
        # debug:
        assert (y_train == self.feeder.train_label).all()
        assert (y_test  == self.feeder.test_label).all()

        # start by the "normal" knn, find for every test sample all its nearest neighbors.
        # we start just with one testing sample
        test_index = 99
        test_image = self.feeder.test_origin_data[test_index]
        test_label = self.feeder.test_label[test_index]
        test_features = X_test_features[test_index]

        # find all its nearest neighbors
        self.knn.fit(X_train_features)
        neighbors = self.knn.kneighbors(test_features)
        print('cool')





