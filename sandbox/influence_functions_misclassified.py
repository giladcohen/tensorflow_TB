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
workspace = 'influence_workspace_misclassified_060319'
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

# get the training features
train_preds_prob, train_features = net.test(X_train, return_embedding=True)
# get the test features
test_preds_prob, test_features = net.test(X_test, return_embedding=True)

# I want to select only from the misclassified test indices
test_preds = np.argmax(test_preds_prob, axis=1)
test_correct = test_preds == y_test
test_misclassified = np.where(test_correct == False)[0]
# choosing 25 misclassified indices in random
test_indices = rand_gen.choice(test_misclassified, 25, replace=False).tolist()

test_features = test_features[test_indices]  # just for these specific test indices

# fit the knn and predict
knn.fit(train_features)
neighbors_indices = knn.kneighbors(test_features, return_distance=False)

# now finding the influence
feeder.reset()

inspector = darkon.Influence(
    workspace=workspace,
    feeder=feeder,
    loss_op_train=net.full_loss,
    loss_op_test=net.loss_op,
    x_placeholder=net.image_placeholder,
    y_placeholder=net.label_placeholder)

testset_batch_size = 100
train_batch_size = 100
train_iterations = 500

approx_params = {
    'scale': 200,
    'num_repeats': 5,
    'recursion_depth': 100,
    'recursion_batch_size': 100
}

# creating the relevant folders
if not os.path.exists(workspace):
    os.makedirs(workspace)

for i, test_index in enumerate(test_indices):
    logger.info("sample {}/{}: calculating scores for test index {}".format(i+1, len(test_indices), test_index))

    # creating the relevant index folder
    dir = os.path.join(workspace, 'test_index_{}'.format(test_index))
    if not os.path.exists(dir):
        os.makedirs(dir)

    scores = inspector.upweighting_influence_batch(
        sess=sess,
        test_indices=[test_index],
        test_batch_size=testset_batch_size,
        approx_params=approx_params,
        train_batch_size=train_batch_size,
        train_iterations=train_iterations)

    sorted_indices = np.argsort(scores)
    harmful = sorted_indices[:50]
    helpful = sorted_indices[-50:][::-1]

    cnt_harmful_in_knn = 0
    print('\nHarmful:')
    for idx in harmful:
        print('[{}] {}'.format(idx, scores[idx]))
        if idx in neighbors_indices[i, 0:50]:
            cnt_harmful_in_knn += 1
    print('{} out of {} harmful images are in the {}-NN'.format(cnt_harmful_in_knn, len(harmful), 50))

    cnt_helpful_in_knn = 0
    print('\nHelpful:')
    for idx in helpful:
        print('[{}] {}'.format(idx, scores[idx]))
        if idx in neighbors_indices[i, 0:50]:
            cnt_helpful_in_knn += 1
    print('{} out of {} helpful images are in the {}-NN'.format(cnt_helpful_in_knn, len(helpful), 50))

    fig, axes1 = plt.subplots(5, 10, figsize=(30, 10))
    target_idx = 0
    for j in range(5):
        for k in range(10):
            idx = neighbors_indices[i, target_idx]
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(feeder.train_origin_data[idx])
            label_str = _classes[int(feeder.train_label[idx])]
            axes1[j][k].set_title('[{}]: {}'.format(idx, label_str))
            target_idx += 1
    plt.savefig(os.path.join(workspace, 'test_index_{}'.format(test_index), 'nearest_neighbors.png'), dpi=350)
    plt.close()

    fig, axes1 = plt.subplots(5, 10, figsize=(30, 10))
    target_idx = 0
    for j in range(5):
        for k in range(10):
            idx = helpful[target_idx]
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(feeder.train_origin_data[idx])
            label_str = _classes[int(feeder.train_label[idx])]
            loc_in_knn = np.where(neighbors_indices[i] == idx)[0][0]
            axes1[j][k].set_title('[{}]: {} #nn:{}'.format(idx, label_str, loc_in_knn))
            target_idx += 1
    plt.savefig(os.path.join(workspace, 'test_index_{}'.format(test_index), 'helpful.png'), dpi=350)
    plt.close()
#
    fig, axes1 = plt.subplots(5, 10, figsize=(30, 10))
    target_idx = 0
    for j in range(5):
        for k in range(10):
            idx = harmful[target_idx]
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(feeder.train_origin_data[idx])
            label_str = _classes[int(feeder.train_label[idx])]
            loc_in_knn = np.where(neighbors_indices[i] == idx)[0][0]
            axes1[j][k].set_title('[{}]: {} #nn:{}'.format(idx, label_str, loc_in_knn))
            target_idx += 1
    plt.savefig(os.path.join(workspace, 'test_index_{}'.format(test_index), 'harmful.png'), dpi=350)
    plt.close()
    print('done')

    # save to disk
    np.save(os.path.join(workspace, 'test_index_{}'.format(test_index), 'scores'), scores)
