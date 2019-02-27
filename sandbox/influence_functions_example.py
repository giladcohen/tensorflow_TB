from darkon_examples.cifar10_resnet.cifar10_train import Train
import darkon_examples.cifar10_resnet.cifar10_input as cifar10_input
import darkon
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

check_point = '/Users/giladcohen/workspace/tensorflow_TB/darkon_examples/cifar10_resnet/pre-trained/model.ckpt-79999'

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

inspector = darkon.Influence(
    workspace='./influence_workspace',
    feeder=feeder,
    loss_op_train=net.full_loss,
    loss_op_test=net.loss_op,
    x_placeholder=net.image_placeholder,
    y_placeholder=net.label_placeholder)

influence_target = 99

# display
print(_classes[int(feeder.test_label[influence_target])])
plt.imshow(feeder.test_origin_data[influence_target])

test_indices = [influence_target]
testset_batch_size = 100

train_batch_size = 100
train_iterations = 500

# train_batch_size = 100
# train_iterations = 50

approx_params = {
    'scale': 200,
    'num_repeats': 5,
    'recursion_depth': 100,
    'recursion_batch_size': 100
}


scores = inspector.upweighting_influence_batch(
    sess,
    test_indices,
    testset_batch_size,
    approx_params,
    train_batch_size,
    train_iterations)

sorted_indices = np.argsort(scores)
harmful = sorted_indices[:10]
helpful = sorted_indices[-10:][::-1]

print('\nHarmful:')
for idx in harmful:
    print('[{}] {}'.format(idx, scores[idx]))

print('\nHelpful:')
for idx in helpful:
    print('[{}] {}'.format(idx, scores[idx]))

fig, axes1 = plt.subplots(2, 5, figsize=(15, 5))
target_idx = 0
for j in range(2):
    for k in range(5):
        idx = helpful[target_idx]
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(feeder.train_origin_data[idx])
        label_str = _classes[int(feeder.train_label[idx])]
        axes1[j][k].set_title('[{}]: {}'.format(idx, label_str))

        target_idx += 1

fig, axes1 = plt.subplots(2, 5, figsize=(15, 5))
target_idx = 0
for j in range(2):
    for k in range(5):
        idx = harmful[target_idx]
        axes1[j][k].set_axis_off()
        axes1[j][k].imshow(feeder.train_origin_data[idx])
        label_str = _classes[int(feeder.train_label[idx])]
        axes1[j][k].set_title('[{}]: {}'.format(idx, label_str))

        target_idx += 1