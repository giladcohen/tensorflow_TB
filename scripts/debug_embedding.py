from __future__ import division

import os
import sys

cwd = os.getcwd() # tensorflow-TB
sys.path.insert(0, cwd)

import lib.logger.logger as logger
from lib.logger.logging_config import logging_config
from utils.parameters import Parameters
from utils.factories import Factories
from lib.datasets.dataset_wrapper import DatasetWrapper
import tensorflow as tf
import numpy as np
from sklearn import manifold
from math import ceil
from utils.misc import *

logging = logging_config()

logging.disable(logging.DEBUG)
log = logger.get_logger('main')

prm_file = '/data/gilad/logs/log_2317_230817_wrn_MOM_simple/parameters.ini'

prm = Parameters()
prm.override(prm_file)

dev = prm.network.DEVICE

factories = Factories(prm)

model = factories.get_model()
model.print_stats()  # debug

preprocessor = factories.get_preprocessor()
preprocessor.print_stats()  # debug

train_dataset = factories.get_train_dataset(preprocessor)
validation_dataset = factories.get_validation_dataset(preprocessor)

dataset_wrapper = DatasetWrapper(prm.dataset.DATASET_NAME + '_wrapper', prm, train_dataset, validation_dataset)
dataset_wrapper.print_stats()

# trainer = factories.get_trainer(model, dataset_wrapper)
# trainer.print_stats()  # debug

# start debugging
model.build_graph()
saver = tf.train.Saver(max_to_keep=None, name='debug', filename='model_debug')
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# restore checkpoint
ckpt_state = tf.train.get_checkpoint_state(prm.train.train_control.CHECKPOINT_DIR)
saver.restore(sess, ckpt_state.model_checkpoint_path)

# get all train/eval features
embedding_dims = 640  # FIXME(gilad): should be: prm.network.NUM_FC_NEURONS
eval_batch_size = 100

def collect_features(dataset):
    """Collecting all the features from the last layer (before the classifier) in the train_set/validation_set"""
    batch_count     = int(ceil(dataset.size / eval_batch_size))
    last_batch_size =          dataset.size % eval_batch_size
    features_vec = -1.0 * np.ones((dataset.size, embedding_dims), dtype=np.float32)
    total_samples = 0  # for debug
    log.info('start storing feature maps for the entire {} set.'.format(str(dataset)))
    dataset.to_preprocess = False  # for train and validation.
    for i in range(batch_count):
        b = i * eval_batch_size
        if i < (batch_count - 1) or (last_batch_size == 0):
            e = (i + 1) * eval_batch_size
        else:
            e = i * eval_batch_size + last_batch_size
        images, labels = dataset.get_mini_batch(indices=range(b, e))
        net = sess.run(model.net, feed_dict={model.images: images,
                                             model.labels: labels,
                                             model.is_training: False})
        features_vec[b:e] = np.reshape(net['embedding_layer'], (e - b, embedding_dims))
        total_samples += images.shape[0]
        log.info('Storing completed: {}%'.format(int(100.0 * e / dataset.size)))

        # debug
        features_tmp = np.array(features_vec[b:e])
        if np.sum(features_tmp == -1) >= eval_batch_size:
            err_str = 'feature_vec equals -1 at least {} times for [b:e]=[{}:{}].'.format(eval_batch_size, b, e)
            print_numpy(features_tmp)
            log.error(err_str)
            raise AssertionError(err_str)

    assert total_samples == dataset.size, \
        'total_samples equals {} instead of {}'.format(total_samples, dataset.size)
    return features_vec

train_features      = collect_features(train_dataset)
train_labels        = train_dataset.labels
validation_features = collect_features(validation_dataset)
validation_labels   = validation_dataset.labels

# computing t-SNE embedding
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
train_embedding_tnse      = tsne.fit_transform(train_features)
validation_embedding_tnse = tsne.fit_transform(validation_features)
plot_embedding(train_embedding_tnse, train_labels, "t-SNE train set embedding")
plot_embedding(validation_embedding_tnse, validation_labels, "t-SNE validation set embedding")

# compute PCA
