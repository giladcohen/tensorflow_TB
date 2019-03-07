from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from darkon_examples.cifar10_resnet.cifar10_train import Train
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
import darkon
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from darkon.log import logger
import os

check_point = 'darkon_examples/cifar10_resnet/pre-trained/model.ckpt-79999'
workspace = 'influence_workspace_060319'
superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# cifar-10 classes
_classes = (
    'airplane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
)

cifar10_input.maybe_download_and_extract()


class MyFeeder(darkon.InfluenceFeeder):
    def __init__(self):
        # load train data
        data, label = cifar10_input.prepare_train_data(padding_size=0)
        self.train_origin_data = data / 256.
        self.train_label = label
        self.train_data = cifar10_input.whitening_image(data)

        # load test data
        data, label = cifar10_input.read_validation_data_wo_whitening()
        self.test_origin_data = data / 256.
        self.test_label = label
        self.test_data = cifar10_input.whitening_image(data)

        self.train_batch_offset = 0

    def test_indices(self, indices):
        return self.test_data[indices], self.test_label[indices]

    def train_batch(self, batch_size):
        # calculate offset
        start = self.train_batch_offset
        end = start + batch_size
        self.train_batch_offset += batch_size

        return self.train_data[start:end, ...], self.train_label[start:end, ...]

    def train_one(self, idx):
        return self.train_data[idx, ...], self.train_label[idx, ...]

    def reset(self):
        self.train_batch_offset = 0


feeder = MyFeeder()

net = Train()
net.build_train_validation_graph()

saver = tf.train.Saver(tf.global_variables())
sess = tf.InteractiveSession()
saver.restore(sess, check_point)

# start the knn observation
knn = NearestNeighbors(n_neighbors=50000, p=2, n_jobs=20)

# get the data
X_train, y_train = feeder.train_batch(50000)
X_test, y_test = feeder.test_indices(range(10000))

# display
# influence_target = 99
# test_indices = [influence_target]
# print(_classes[int(feeder.test_label[influence_target])])
# plt.imshow(feeder.test_origin_data[influence_target])

test_indices = []
for cls in range(len(_classes)):
    cls_test_indices = rand_gen.choice(np.where(y_test==cls)[0], 5, replace=False).tolist()
    test_indices.extend(cls_test_indices)

# get the training features
train_preds_prob, train_features = net.test(X_train, return_embedding=True)
# get the test features
test_preds_prob, test_features = net.test(X_test, return_embedding=True)

test_features_1 = test_features[test_indices]  # just for these specific test indices

# fit the knn and predict
knn.fit(train_features)
neighbors_indices_1 = knn.kneighbors(test_features_1, return_distance=False)

tot_loc_help_sum = np.zeros(shape=(len(test_indices)))
tot_loc_harm_sum = np.zeros(shape=(len(test_indices)))
for i, test_index in enumerate(test_indices[0:12]):
    # loading the scores
    scores = np.load('influence_workspace_060319/test_index_{}/scores.npy'.format(test_index))
    sorted_indices = np.argsort(scores)
    harmful = sorted_indices[:50]
    helpful = sorted_indices[-50:][::-1]

    for idx in helpful:
        loc_in_knn = np.where(neighbors_indices_1[i] == idx)[0][0]
        tot_loc_help_sum[i] += loc_in_knn
    for idx in harmful:
        loc_in_knn = np.where(neighbors_indices_1[i] == idx)[0][0]
        tot_loc_harm_sum[i] += loc_in_knn

test_indices_misclassified = [2372, 1353, 9791, 2719, 1181, 8577, 4192, 5518, 9734, 3995, 4152, 47, 6833, 9041, 6271]
test_features_2 = test_features[test_indices_misclassified]
neighbors_indices_2 = knn.kneighbors(test_features_2, return_distance=False)

tot_loc_help_sum_2 = np.zeros(shape=(len(test_indices_misclassified)))
tot_loc_harm_sum_2 = np.zeros(shape=(len(test_indices_misclassified)))
for i, test_index in enumerate(test_indices_misclassified):
    # loading the scores
    scores = np.load('influence_workspace_misclassified_060319/test_index_{}/scores.npy'.format(test_index))
    sorted_indices = np.argsort(scores)
    harmful = sorted_indices[:50]
    helpful = sorted_indices[-50:][::-1]

    for idx in helpful:
        loc_in_knn = np.where(neighbors_indices_2[i] == idx)[0][0]
        tot_loc_help_sum_2[i] += loc_in_knn
    for idx in harmful:
        loc_in_knn = np.where(neighbors_indices_2[i] == idx)[0][0]
        tot_loc_harm_sum_2[i] += loc_in_knn
