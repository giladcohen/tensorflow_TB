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

check_point = 'darkon_examples/cifar10_resnet/pre-trained/model.ckpt-79999'
superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# cifar-10 classes
_classes = (
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
train_features = -1000 * np.ones((50000, 64))
for i in np.arange(500):
    begin = 100 * i
    end = 100 * (i+1)
    train_features[begin:end] = sess.run(net.embedding_op, feed_dict={net.flexi_image_placeholder: X_train[begin:end]})
# get the testing features
test_features = -1000 * np.ones((10000, 64))
for i in np.arange(100):
    begin = 100 * i
    end = 100 * (i+1)
    test_features[begin:end] = sess.run(net.embedding_op, feed_dict={net.flexi_image_placeholder: X_test[begin:end]})

assert np.sum(train_features == -1000) == 0
assert np.sum(test_features == -1000) == 0

test_features = test_features[test_indices]  # just for these specific test indices

# fit the knn and predict
knn.fit(train_features)
neighbors_indices = knn.kneighbors(test_features, return_distance=False)
# neighbors_labels = []
# for i in range(50):
#     neighbors_labels.append(_classes[int(y_train[neighbors_indices[0, i]])])
#
# # printing:
# print('neighbors_indices={}'.format(neighbors_indices))
# print('neighbors_labels={}'.format(neighbors_labels))

# now finding the influence
feeder.reset()

inspector = darkon.Influence(
    workspace='./influence_workspace',
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

for i, test_index in enumerate(test_indices):
    logger.info("sample {}/{}: calculating scores for test index {}".format(i+1, len(test_indices), test_index))
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
    plt.savefig('./influence_workspace/nearest_for_test_index_{}.png'.format(test_index), dpi=350)
    plt.clf()

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
    plt.savefig('./influence_workspace/helpful_for_test_index_{}.png'.format(test_index), dpi=350)
    plt.clf()
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
    plt.savefig('./influence_workspace/harmful_for_test_index_{}.png'.format(test_index), dpi=350)
    plt.clf()
    print('done')

    # save to disk
    np.save('./influence_workspace/scores_for_test_index_{}.npy'.format(test_index), scores)

print('done')
