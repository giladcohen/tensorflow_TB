
"""ResNet Train/Eval module. This code used to calculate t-SNE
"""
import time
import six
import sys

import cifar_input1
import numpy as np
import resnet_model
import tensorflow as tf
import os.path
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('eval_dir', '', 'Directory to keep eval outputs.')

BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
LOG_ROOT = os.path.join(BASE_PATH, 'logs_wrn28-10_1513_220317')
TRAIN_SET_SIZE = 50000
TEST_SET_SIZE  = 10000
BATCH_SIZE=100
TRAIN_BATCH_COUNT  = TRAIN_SET_SIZE/BATCH_SIZE
EVAL_BATCH_COUNT   = TEST_SET_SIZE/BATCH_SIZE
FLAGS.eval_dir = os.path.join(BASE_PATH, 'logs_wrn28-10_1513_220317/test')

hps = resnet_model.HParams(batch_size=BATCH_SIZE,
                             num_classes=10,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=4,
                             use_bottleneck=False,
                             weight_decay_rate=0.0005,
                             relu_leakiness=0.1,
                             optimizer='mom',
                             use_nesterov=True)

def plot_embedding(X, Y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(Y[i]),
                 color=plt.cm.Set1(Y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

train_images     = np.empty((TRAIN_SET_SIZE,32,32,3))
train_labels_vec = -1 * np.ones(TEST_SET_SIZE, dtype=np.int)
train_pool_vec   = -1 * np.ones((TEST_SET_SIZE, 640), dtype=np.float32)
test_images      = np.empty((TRAIN_SET_SIZE,32,32,3))
test_labels_vec  = -1 * np.ones(TEST_SET_SIZE, dtype=np.int)
test_pool_vec    = -1 * np.ones((TEST_SET_SIZE, 640), dtype=np.float32)

images, labels = cifar_input1.build_input('cifar10',
                    os.path.join(BASE_PATH,'cifar10/*.bin'), BATCH_SIZE, 'eval') #includes both train and test

model = resnet_model.ResNet(hps, images, labels, 'eval')
model.build_graph()
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU': 0}))

tf.train.start_queue_runners(sess)

try:
    ckpt_state = tf.train.get_checkpoint_state(LOG_ROOT)
except tf.errors.OutOfRangeError as e:
    tf.logging.error('Cannot restore checkpoint: %s', e)
if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('No model to eval yet at %s', LOG_ROOT)
tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
saver.restore(sess, ckpt_state.model_checkpoint_path)

total_prediction, correct_prediction = 0, 0
for i in six.moves.range(TRAIN_BATCH_COUNT):
    (summaries, loss, imgs, pool, predictions, truth, train_step) = sess.run(
                                                        [model.summaries, model.cost, 
                                                         model._images, model.pool_out, model.predictions,
                                                         model.labels, model.global_step])
    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]
    tmp_prediction = 1.0 * correct_prediction / total_prediction
    train_images[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = imgs
    train_labels_vec[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = truth
    train_pool_vec[i*BATCH_SIZE:(i+1)*BATCH_SIZE]  = np.reshape(pool, (BATCH_SIZE, -1))
    print("run train batch %0d out of %0d. tmp_prediction = %.4f" %(i+1, TRAIN_SET_SIZE/BATCH_SIZE, tmp_prediction))
 
precision = 1.0 * correct_prediction / total_prediction
print('For train set: loss: %.3f, precision: %.3f' % (loss, precision))
tf.logging.info('For train set: loss: %.3f, precision: %.3f' % (loss, precision))

total_prediction, correct_prediction = 0, 0
for i in six.moves.range(EVAL_BATCH_COUNT):
    (summaries, loss, imgs, pool, predictions, truth, train_step) = sess.run(
                                                        [model.summaries, model.cost, 
                                                         model._images, model.pool_out, model.predictions,
                                                         model.labels, model.global_step])
    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]
    tmp_prediction = 1.0 * correct_prediction / total_prediction
    train_images[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = imgs
    test_labels_vec[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = truth
    test_pool_vec[i*BATCH_SIZE:(i+1)*BATCH_SIZE]  = np.reshape(pool, (BATCH_SIZE, -1))
    print("run test batch %0d out of %0d. tmp_prediction = %.4f" %(i+1, TEST_SET_SIZE/BATCH_SIZE, tmp_prediction))
 
precision = 1.0 * correct_prediction / total_prediction
print('For test set: loss: %.3f, precision: %.3f' % (loss, precision))
tf.logging.info('For test set: loss: %.3f, precision: %.3f' % (loss, precision))



# plotting t-SNE
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
pool_vec_tsne = tsne.fit_transorm(pool_vec)
# plot_embedding(pool_vec_tsne, labels_vec, "t-SNE embedding of the average pool layer output")
knn = cv2.ml.KNearest_create()
print ('Start training kNN model...')
knn.train(pool_vec, cv2.ml.ROW_SAMPLE, labels_vec)
print ('Done training kNN model.')
k_arr    = np.array([1, 3, 5, 7, 9, 11, 31, 51, 101, 151, 201, 301, 401, 501])
# for k in np.nditer(k_arr):
#     ret, results, neighbors, dist = knn.findNearest(Y, k)
#     acc = np.mean(results == test_labels)
#     print("k=%0d: %.4f" %(k, acc))

print ("KNN accuracies:")



