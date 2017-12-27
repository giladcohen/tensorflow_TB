'''This code converts a numpy image to .bin in the same format of cifar10'''
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
from keras.datasets import cifar10, cifar100
import cv2
import os
import contextlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from math import ceil

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

def save_dataset_to_disk(dataset_name, train_data_dir, train_labels_file, test_data_dir, test_labels_file):
    """Saving CIFAR10/100 train/test data to specified dirs
       Saving CIFAR10/100 train/test labels to specified files"""
    if 'cifar100' in dataset_name:
        dataset = cifar100
    elif 'cifar10' in dataset_name:
        dataset = cifar10
    else:
        raise AssertionError('dataset {} is not supported'.format(dataset_name))

    (X_train, Y_train), (X_test, Y_test) = dataset.load_data()
    np.savetxt(train_labels_file, Y_train, fmt='%0d')
    np.savetxt(test_labels_file,  Y_test,  fmt='%0d')
    for i in range(X_train.shape[0]):
        img = X_train[i]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(train_data_dir, 'train_image_%0d.png' % i), img_bgr)
    for i in range(X_test.shape[0]):
        img = X_test[i]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(test_data_dir,  'test_image_%0d.png'  % i), img_bgr)

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

def plot_embedding2(vis_x, vis_y, c, title=None):
    plt.figure()
    plt.scatter(vis_x, vis_y, c=c, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    # plt.colorbar(ticks=['airplane', 'automobile', 'bird',
    #                     'cat', 'deer', 'dog', 'frog', 'horse',
    #                     'ship', 'truck'])
    plt.clim(-0.5, 9.5)
    if title is not None:
        plt.title(title)
    plt.show()

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

def collect_features(agent, dataset_type, fetches, feed_dict=None):
        """Collecting all fetches from the DNN in the dataset (train/validation/test)
        :param agent: The agent (trainer/tester).
                      Must have a session (sess), batch size (eval_batch_size), logger (log) and dataset wrapper (dataset)
                      The agent must have a model with images and labels.  # This should be updated for all models
        :param dataset_type: 'train' or 'validation'
        :param fetches: list of all the fetches to sample from the DNN.
        :param feed_dict: feed_dict to sess.run, other than images/labels/is_training.
        :return: fetches, as numpy float32.
        """
        if feed_dict is None:
            feed_dict = {}

        batch_size = agent.eval_batch_size
        log        = agent.log
        sess       = agent.sess
        model      = agent.model

        if dataset_type == 'train':
            dataset = agent.dataset.train_dataset
        elif dataset_type == 'validation':
            dataset = agent.dataset.validation_dataset
        else:
            err_str = 'dataset_type={} is not supported'.format(dataset_type)
            log.error(err_str)
            raise AssertionError(err_str)
        dataset.to_preprocess = False

        fetches_dims = [(batch_size,) + tuple(fetches[i].get_shape().as_list()[1:]) for i in xrange(len(fetches))]

        batch_count     = int(ceil(dataset.size / batch_size))
        last_batch_size =          dataset.size % batch_size
        fetches_np = [np.empty(fetches_dims[i], dtype=np.float32) for i in xrange(len(fetches))]

        log.info('start storing fetches for the entire {} set.'.format(str(dataset)))
        for i in range(batch_count):
            b = i * batch_size
            if i < (batch_count - 1) or (last_batch_size == 0):
                e = (i + 1) * batch_size
            else:
                e = i * batch_size + last_batch_size
            images, labels = dataset.get_mini_batch(indices=range(b, e))
            tmp_feed_dict = {model.images: images,
                             model.labels: labels,
                             model.is_training: False}
            tmp_feed_dict.update(feed_dict)
            fetches_out = sess.run(fetches=fetches, feed_dict=tmp_feed_dict)
            for i in xrange(len(fetches)):
                fetches_np[i][b:e] = np.reshape(fetches_out[i], (e - b, ) + fetches_dims[i][1:])
            log.info('Storing completed: {}%'.format(int(100.0 * e / dataset.size)))

        if dataset_type == 'train':
            dataset.to_preprocess = True

        return tuple(fetches_np)
