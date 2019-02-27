from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.testers.knn_classifier_tester import KNNClassifierTester
from sklearn.neighbors import NearestNeighbors
from lib.datasets.influence_feeder import Feeder
import darkon
import matplotlib.pyplot as plt

class KNNInfluenceTester(KNNClassifierTester):

    def __init__(self, *args, **kwargs):
        super(KNNInfluenceTester, self).__init__(*args, **kwargs)

        self.knn = NearestNeighbors(n_neighbors=self.knn_neighbors,  # should be the number of the training set
                                    # algorithm='brute',
                                    p=int(self.knn_norm[-1]),
                                    n_jobs=self.knn_jobs)

        self.feeder = Feeder('influence_feeder', self.prm)

        self.one_hot_labels = self.prm.network.ONE_HOT_LABELS

        self._classes = (
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck'
        )

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
        neighbors_indices = self.knn.kneighbors(np.expand_dims(test_features, axis=0), return_distance=False)
        print('neighbors_indices={}'.format(neighbors_indices))

        # now find the influence
        inspector = darkon.Influence(
            workspace='./influence_workspace',
            feeder=self.feeder,
            loss_op_train=self.model.cost,
            loss_op_test=self.model.cost,
            x_placeholder=self.model.images,
            y_placeholder=self.model.labels)

        approx_params = {
            'scale': 200,
            'num_repeats': 5,
            'recursion_depth': 100,
            'recursion_batch_size': 100
        }

        scores = inspector.upweighting_influence_batch(
            sess=self.sess,
            test_indices=[test_index],
            test_batch_size=self.prm.train.train_control.EVAL_BATCH_SIZE,
            approx_params=approx_params,
            train_batch_size=self.prm.train.train_control.TRAIN_BATCH_SIZE,
            train_iterations=500)

        sorted_indices = np.argsort(scores)
        harmful = sorted_indices[:50]
        helpful = sorted_indices[-50:][::-1]

        cnt_harmful_in_knn = 0
        print('\nHarmful:')
        for idx in harmful:
            print('[{}] {}'.format(idx, scores[idx]))
            if idx in neighbors_indices:
                cnt_harmful_in_knn += 1
        print('{} out of {} harmful images are in the {}-NN'.format(cnt_harmful_in_knn, len(harmful), self.knn_neighbors))

        cnt_helpful_in_knn = 0
        print('\nHelpful:')
        for idx in helpful:
            print('[{}] {}'.format(idx, scores[idx]))
            if idx in neighbors_indices:
                cnt_helpful_in_knn += 1
        print('{} out of {} helpful images are in the {}-NN'.format(cnt_helpful_in_knn, len(helpful), self.knn_neighbors))

        fig, axes1 = plt.subplots(5, 10, figsize=(15, 5))
        target_idx = 0
        for j in range(5):
            for k in range(10):
                idx = helpful[target_idx]
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(self.feeder.train_origin_data[idx])
                if self.one_hot_labels:
                    label = np.argmax(self.feeder.train_label[idx])
                else:
                    label = self.feeder.train_label[idx]
                label_str = self._classes[int(label)]
                axes1[j][k].set_title('[{}]: {}'.format(idx, label_str))

                target_idx += 1
        plt.savefig('./influence_workspace/helpful.png', dpi=350)
        plt.clf()

        fig, axes1 = plt.subplots(5, 10, figsize=(15, 5))
        target_idx = 0
        for j in range(5):
            for k in range(10):
                idx = harmful[target_idx]
                axes1[j][k].set_axis_off()
                axes1[j][k].imshow(self.feeder.train_origin_data[idx])
                if self.one_hot_labels:
                    label = np.argmax(self.feeder.train_label[idx])
                else:
                    label = self.feeder.train_label[idx]
                label_str = self._classes[int(label)]
                axes1[j][k].set_title('[{}]: {}'.format(idx, label_str))

                target_idx += 1
        plt.savefig('./influence_workspace/harmful.png', dpi=350)
        plt.clf()
        print('done')
