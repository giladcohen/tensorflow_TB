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

        # start by the "normal" knn, find for every test sample all its nearest neighbors.
        # we start just with one testing sample
        test_index = 99
        test_image = self.feeder.test_origin_data[test_index]
        test_label = self.feeder.test_label[test_index]
        plt.imshow(test_image)



