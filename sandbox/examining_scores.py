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
from scipy.stats import norm

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

test_indices_1 = []
for cls in range(len(_classes)):
    cls_test_indices = rand_gen.choice(np.where(y_test==cls)[0], 5, replace=False).tolist()
    test_indices_1.extend(cls_test_indices)

# get the training features
train_preds_prob, train_features = net.test(X_train, return_embedding=True)
# get the test features
test_preds_prob, test_features = net.test(X_test, return_embedding=True)

test_features_1 = test_features[test_indices_1]  # just for these specific test indices

# fit the knn and predict
knn.fit(train_features)
neighbors_indices_1 = knn.kneighbors(test_features_1, return_distance=False)

tot_loc_help_sum = np.zeros(shape=(len(test_indices_1)))
tot_loc_harm_sum = np.zeros(shape=(len(test_indices_1)))
for i, test_index in enumerate(test_indices_1):
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

avg_loc_help = tot_loc_help_sum / 50
avg_loc_harm = tot_loc_harm_sum / 50

test_indices_2 = [2216, 8932, 2136,  138, 6900, 1746, 5330,   47, 3753, 2719,
                  1181, 6833, 6772, 5871, 6271, 8295, 7813, 2001, 7370, 5518,
                  7861, 4012,   57, 8757, 4833, 9734, 4954, 3787, 8577, 4192,
                  2486, 1884, 3231, 5013,  893, 2435, 4055, 4097, 2372, 9041,
                  6741, 4152, 8549, 3995, 1353, 6014, 5506, 8446, 9791]

test_features_2 = test_features[test_indices_2]
neighbors_indices_2 = knn.kneighbors(test_features_2, return_distance=False)

tot_loc_help_sum_2 = np.zeros(shape=(len(test_indices_2)))
tot_loc_harm_sum_2 = np.zeros(shape=(len(test_indices_2)))
for i, test_index in enumerate(test_indices_2):
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

avg_loc_help_2 = tot_loc_help_sum_2 / 50
avg_loc_harm_2 = tot_loc_harm_sum_2 / 50

# Assuming gaussian distribution. Calculation mean and std for avg_loc_help
avg_loc_help_mean   = np.mean(avg_loc_help)
avg_loc_help_std    = np.std(avg_loc_help, ddof=1)
avg_loc_help_2_mean = np.mean(avg_loc_help_2)
avg_loc_help_2_std  = np.std(avg_loc_help_2, ddof=1)

avg_loc_harm_mean   = np.mean(avg_loc_harm)
avg_loc_harm_std    = np.std(avg_loc_harm, ddof=1)
avg_loc_harm_2_mean = np.mean(avg_loc_harm_2)
avg_loc_harm_2_std  = np.std(avg_loc_harm_2, ddof=1)

# drawing the two gaussians

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

m2   = avg_loc_harm_mean
std2 = avg_loc_harm_std
m1   = avg_loc_harm_2_mean
std1 = avg_loc_harm_2_std

#Get point of intersect
result = solve(m1, m2, std1, std2)

#Get point on surface
x = np.linspace(-10000, 50000, 50000)
plot1 = plt.plot(x, norm.pdf(x, m1, std1))
plot2 = plt.plot(x, norm.pdf(x, m2, std2))
plot3 = plt.plot(result[1], norm.pdf(result[1], m1, std1), 'o')

#Plots integrated area
r = result[1]
olap = plt.fill_between(x[x>r], 0, norm.pdf(x[x>r], m1, std1), alpha=0.3)
olap = plt.fill_between(x[x<r], 0, norm.pdf(x[x<r], m2, std2), alpha=0.3)

# integrate
area = norm.cdf(r, m2, std2) + (1. -norm.cdf(r, m1, std1))
print("Area under curves ", area)

plt.show()