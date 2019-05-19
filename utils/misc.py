'''This code converts a numpy image to .bin in the same format of cifar10'''
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import cv2
import os
import contextlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from math import ceil
import time
import datetime
import re

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def convert_numpy_to_bin(images, labels, save_file, h=32, w=32):
    """Converts numpy data in the form:
    images: [N, H, W, D]
    labels: [N]
    to a .bin file in a CIFAR10 protocol
    """
    images = (np.array(images))
    N = images.shape[0]
    record_bytes = 3 * h * w + 1 #includes also the label
    out = np.zeros([record_bytes * N], np.uint8)
    for i in range(N):
        im = images[i]
        r = im[:,:,0].flatten()
        g = im[:,:,1].flatten()
        b = im[:,:,2].flatten()
        label = labels[i]
        out[i*record_bytes:(i+1)*record_bytes] = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
    out.tofile(save_file)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def print_numpy(arr):
    """
    :param arr: numpy array
    :return: no return
    """
    @contextlib.contextmanager
    def printoptions(*args, **kwargs):
        original = np.get_printoptions()
        np.set_printoptions(*args, **kwargs)
        try:
            yield
        finally:
            np.set_printoptions(**original)

    with printoptions(precision=3, suppress=True, formatter={'float': '{: 0.3f}'.format}):
        print(arr)

def get_plain_session(sess):
    """
    Bypassing tensorflow issue:
    https://github.com/tensorflow/tensorflow/issues/8425
    :param sess: Monitored session
    :return: Session object
    """
    session = sess
    while type(session).__name__ != 'Session':
        session = session._sess
    return session

def np_evaluate(sess, fetches, x_set, y_set, x, y, batch_size, feed_dict=None, log=None, y_adv=None, y_adv_set=None):
    """
    Collecting tensorfow operators in a numpy API
    :param sess: tf.Session
    :param fetches: list of all the tf tensors (tf.Tensor) to collect
    :param x_set: dataset in np.ndarray
    :param y_set: dataset labels in np.ndarray
    :param x: input placeholder (tf.placeholder) with size of [? , x_set.shape[1:]]
    :param y: label placeholder (tf.placeholder). one hot
    :param batch_size: batch size to process
    :param feed_dict: default feed_dict (dictionary) to apply in sess.run calls
    :param log: logger
    :param y_adv: adversarial placeholder (tf.placeholder)
    :param y_adv_set: adversarial data in np.array
    :return: fetches as np.ndarray.
    """

    if feed_dict is None:
        feed_dict = {}

    num_samples = len(x_set)
    batch_count = int(ceil(num_samples / batch_size))
    last_batch_size = num_samples % batch_size

    fetches_dims = [(num_samples,) + tuple(fetches[i].get_shape().as_list()[1:]) for i in xrange(len(fetches))]
    fetches_np = [np.empty(fetches_dims[i], dtype=np.float32) for i in xrange(len(fetches))]

    for i in range(batch_count):
        b = i * batch_size
        if i < (batch_count - 1) or (last_batch_size == 0):
            e = (i + 1) * batch_size
        else:
            e = i * batch_size + last_batch_size
        images = x_set[b:e]
        labels = y_set[b:e]
        tmp_feed_dict = {x: images, y: labels}
        if y_adv is not None:
            adv_labels = y_adv_set[b:e]
            tmp_feed_dict.update({y_adv: adv_labels})
        tmp_feed_dict.update(feed_dict)
        fetches_out = sess.run(fetches=fetches, feed_dict=tmp_feed_dict)
        for i in xrange(len(fetches)):
            fetches_np[i][b:e] = np.reshape(fetches_out[i], (e - b,) + fetches_dims[i][1:])
        progress_str = 'Storing completed: {}%'.format(int(100.0 * e / num_samples))
        if log is None:
            print(progress_str)
        else:
            log.info(progress_str)
    return tuple(fetches_np)

def collect_features(agent, dataset_name, fetches, feed_dict=None):
    """Collecting all fetches from the DNN in the dataset (train/validation/test/train_eval)
    :param agent: The agent (trainer/tester).
                  Must have a session (sess), batch size (eval_batch_size), logger (log) and dataset wrapper (dataset)
                  The agent must have a model with images and labels.  # This should be updated for all models
    :param dataset_name: 'train', 'validation' or "test"
    :param fetches: list of all the fetches to sample from the DNN.
    :param feed_dict: feed_dict to sess.run, other than images/labels/is_training.
    :return: fetches, as numpy float32.
    """
    if feed_dict is None:
        feed_dict = {}

    batch_size = agent.eval_batch_size
    log        = agent.log
    model      = agent.model
    dataset    = agent.dataset
    sess       = agent.plain_sess

    if dataset_name == 'train_eval':
        num_samples = dataset.train_set_size
        sess.run(dataset.train_eval_iterator.initializer)
    elif dataset_name == 'train_pool_eval':
        num_samples = dataset.pool_size
        sess.run(dataset.train_pool_eval_iterator.initializer)
    elif dataset_name == 'train_unpool_eval':
        num_samples = dataset.unpool_size
        sess.run(dataset.train_unpool_eval_iterator.initializer)
    elif dataset_name == 'train_random_eval':
        num_samples = dataset.train_set_size
        sess.run(dataset.train_random_eval_iterator.initializer)
    elif dataset_name == 'validation':
        num_samples = dataset.validation_set_size
        sess.run(dataset.validation_iterator.initializer)
    elif dataset_name == 'test':
        num_samples = dataset.test_set_size
        sess.run(dataset.test_iterator.initializer)
    else:
        err_str = 'dataset_name={} is not supported'.format(dataset_name)
        log.error(err_str)
        raise AssertionError(err_str)

    fetches_dims = [(num_samples,) + tuple(fetches[i].get_shape().as_list()[1:]) for i in xrange(len(fetches))]

    batch_count     = int(ceil(num_samples / batch_size))
    last_batch_size =          num_samples % batch_size
    fetches_np = [np.empty(fetches_dims[i], dtype=np.float32) for i in xrange(len(fetches))]

    log.info('start storing 2d fetches for {} samples in the {} set.'.format(num_samples, dataset_name))
    for i in range(batch_count):
        b = i * batch_size
        if i < (batch_count - 1) or (last_batch_size == 0):
            e = (i + 1) * batch_size
        else:
            e = i * batch_size + last_batch_size
        _, images, labels = dataset.get_mini_batch(dataset_name, sess)
        tmp_feed_dict = {model.images: images,
                         model.labels: labels,
                         model.is_training: False}
        tmp_feed_dict.update(feed_dict)
        fetches_out = sess.run(fetches=fetches, feed_dict=tmp_feed_dict)
        for i in xrange(len(fetches)):
            fetches_np[i][b:e] = np.reshape(fetches_out[i], (e - b,) + fetches_dims[i][1:])
        log.info('Storing completed: {}%'.format(int(100.0 * e / num_samples)))

    return tuple(fetches_np)

def collect_features_1d(agent, dataset_name, fetches, feed_dict=None):
    """Collecting all fetches from the DNN in the dataset (train/validation/test/train_eval)
    This function supports aggregation and averaging of 1d signals (scalelr per minibatch) in the network
    :param agent: The agent (trainer/tester).
                  Must have a session (sess), batch size (eval_batch_size), logger (log) and dataset wrapper (dataset)
                  The agent must have a model with images and labels.  # This should be updated for all models
    :param dataset_name: 'train' or 'validation'
    :param fetches: list of all the fetches to sample from the DNN.
    :param feed_dict: feed_dict to sess.run, other than images/labels/is_training.
    :return: fetches, as numpy float32.
    """

    if feed_dict is None:
        feed_dict = {}

    batch_size = agent.eval_batch_size
    log        = agent.log
    model      = agent.model
    dataset    = agent.dataset
    sess       = agent.plain_sess

    if dataset_name == 'train':
        num_samples = dataset.train_set_size
    elif dataset_name == 'train_pool':
        num_samples = dataset.pool_size
    elif dataset_name == 'train_eval':
        num_samples = dataset.train_set_size
        sess.run(dataset.train_eval_iterator.initializer)
    elif dataset_name == 'train_pool_eval':
        num_samples = dataset.pool_size
        sess.run(dataset.train_pool_eval_iterator.initializer)
    elif dataset_name == 'train_random_eval':
        num_samples = dataset.train_set_size
        sess.run(dataset.train_random_eval_iterator.initializer)
    elif dataset_name == 'validation':
        num_samples = dataset.validation_set_size
        sess.run(dataset.validation_iterator.initializer)
    elif dataset_name == 'test':
        num_samples = dataset.test_set_size
        sess.run(dataset.test_iterator.initializer)
    else:
        err_str = 'dataset_name={} is not supported'.format(dataset_name)
        log.error(err_str)
        raise AssertionError(err_str)

    batch_count     = int(ceil(num_samples / batch_size))
    last_batch_size =          num_samples % batch_size

    total_fetches_np = np.zeros(shape=(len(fetches)), dtype=np.float32)

    log.info('start storing 1d fetches for {} samples in the {} set.'.format(num_samples, dataset_name))
    for i in range(batch_count):
        b = i * batch_size
        if i < (batch_count - 1) or (last_batch_size == 0):
            e = (i + 1) * batch_size
        else:
            e = i * batch_size + last_batch_size
        _, images, labels = dataset.get_mini_batch(dataset_name, sess)
        tmp_feed_dict = {model.images: images,
                         model.labels: labels,
                         model.is_training: False}
        tmp_feed_dict.update(feed_dict)
        fetches_out = sess.run(fetches=fetches, feed_dict=tmp_feed_dict)
        for i in xrange(len(fetches)):
            total_fetches_np[i] += fetches_out[i] * (e - b)
        log.info('Storing completed: {}%'.format(int(100.0 * e / num_samples)))

    fetches_np = total_fetches_np / num_samples

    return tuple(fetches_np)

def get_vars(all_vars, *var_patt):
    """
    get all vars of model.
    common usage for all_vars: all_vars=tf.global_variables()
    var_patt: exclude of vars with specific pattern.
    """
    vars       = []
    other_vars = []
    for v in all_vars:
        found = False
        for var in var_patt:
            if var in v.name:
                found = True
        if found:
            vars.append(v)
        else:
            other_vars.append(v)
    return vars, other_vars

def corr_distance(x, y):
    """Correlation distance between x and y"""
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    ret = 1 - (np.dot(x, y) / (norm_x * norm_y))
    assert 0 <= ret <= 1
    return ret

def get_timestamp():
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')

def calc_mutual_agreement(y1, y2, y_gt):
    """
    This function gets to prediction vectors y1, y2 and calculates the
    mutual agreement and disagreement scores
    MA = (DNN right and KNN right)/(DNN right)
    MD = (DNN wrong and KNN wrong like the DNN)/(DNN wrong)
    :param y1: DNN prediction vector
    :param y2: KNN prediction vector
    :param y_gt: Ground truth labels
    :return: MA and MD scores
    """
    assert y1.shape == y2.shape == y_gt.shape, "labels' shape do not match"
    dnn_correct_cnt = dnn_wrong_cnt = 0
    ma_cnt = md_cnt = 0

    for i in xrange(y_gt.shape[0]):
        if y1[i] == y_gt[i]:
            dnn_correct_cnt += 1
            if y2[i] == y1[i]:
                ma_cnt += 1
        else:
            dnn_wrong_cnt += 1
            if y2[i] == y1[i]:
                md_cnt += 1

    if dnn_correct_cnt > 0:
        ma_score = ma_cnt / dnn_correct_cnt
    else:
        ma_score = -1.0  # pevent dividing by 0

    if dnn_wrong_cnt > 0:
        md_score = md_cnt / dnn_wrong_cnt
    else:
        md_score = -1.0  # pevent dividing by 0
    return ma_score, md_score

def calc_psame(y1, y2):
    """
    Calculating P_SAME
    :param y1: DNN prediction
    :param y2: SVM/k-NN/other prediction
    :param y_gt: ground truth
    :return: P_SAME value
    """
    assert y1.shape == y2.shape, "labels' shape do not match"
    match_cnt = 0
    for i in xrange(y1.shape[0]):
        if y1[i] == y2[i]:
            match_cnt += 1
    p_same = match_cnt / y1.shape[0]
    return p_same

def one_hot(indices, depth):
    """Converting the indices to one hot representation
    :param indices: numpy array
    :param depth: the depth of the one hot vectors
    """
    ohm = np.zeros([indices.shape[0], depth])
    ohm[np.arange(indices.shape[0]), indices] = 1
    return ohm

def err_n_assert(logger, s):
    logger.error(s)
    raise AssertionError(s)

