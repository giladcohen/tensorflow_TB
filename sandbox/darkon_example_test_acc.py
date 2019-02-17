from darkon_examples.cifar10_resnet.cifar10_train import Train
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
import darkon
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

check_point = '/Users/giladcohen/workspace/tensorflow-TB/darkon_examples/cifar10_resnet/pre-trained/model.ckpt-79999'

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
# net.build_train_validation_graph()
#
# saver = tf.train.Saver(tf.global_variables())
# sess = tf.InteractiveSession()
# saver.restore(sess, check_point)
#
# inspector = darkon.Influence(
#     workspace='./influence_workspace',
#     feeder=feeder,
#     loss_op_train=net.full_loss,
#     loss_op_test=net.loss_op,
#     x_placeholder=net.image_placeholder,
#     y_placeholder=net.label_placeholder)
#
# influence_target = 99
#
# # display
# print(_classes[int(feeder.test_label[influence_target])])
# plt.imshow(feeder.test_origin_data[influence_target])
#
# test_indices = [influence_target]
# testset_batch_size = 100
#
# train_batch_size = 100
# train_iterations = 500

# train_batch_size = 100
# train_iterations = 50

# batch_indices = range(32)
# batch_images, batch_labels = feeder.test_indices(batch_indices)
# batch_pred = sess.run(net.)

test_indices = range(10000)
test_images, test_labels = feeder.test_indices(test_indices)
test_pred = net.test(test_images)
test_pred_2 = np.argmax(test_pred, axis=1)

test_acc = np.sum(test_pred_2 == test_labels) / 10000.0