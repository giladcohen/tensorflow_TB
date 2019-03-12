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

# get the training features
train_preds_prob, train_features = net.test(X_train, return_embedding=True)
# get the test features
test_preds_prob, test_features = net.test(X_test, return_embedding=True)

# fit the knn
knn.fit(train_features)

def get_knn(test_index):
    """
    :param test_indice: test index to return all its neighbors (i.e. 99)
    :return: all neighbors. np.ndarray with shape (50000,)
    """
    # first get all the test features
    test_features_i = test_features[test_index]
