from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import numpy as np


def process_image(img_path, image_format, image_means, target_height, target_width):
    """
    tensorflow function
    :param img_path: path to file containing an image
    :return: image - Tensor image after preprocessing (scaling and possibly mean substraction
             [scale_x,scale_y] - Tensors of scales [target_w/source_w, target_h/source_h]
    """
    img_file = tf.read_file(img_path)
    image = tf.image.decode_image(img_file, channels=3)
    image.set_shape([None, None, 3])
    source_h = tf.shape(image)[0]
    source_w = tf.shape(image)[1]

    if image_format == 'BGR':
        channels = tf.unstack(image, num=3, axis=-1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=-1)

    # pre-processing of image
    image = tf.image.resize_images(image, [target_height, target_width])
    scale_y = tf.to_double(tf.shape(image)[0] / source_h)
    scale_x = tf.to_double(tf.shape(image)[1] / source_w)

    if image_means is not None:
        image = tf.subtract(image, image_means)

    return image, [scale_x, scale_y]

def identity(self, image, labels):
    return image, labels


def flip(image, labels):
    image = tf.image.flip_left_right(image)

    wx = tf.to_double(tf.shape(image)[1])
    bboxes = tf.stack([wx - labels[:, 0], labels[:, 1], wx - labels[:, 2], labels[:, 3]], axis=-1)

    return image, bboxes

def random_flip(image, labels, flip_prob):
    r = tf.random_uniform(shape=())
    to_flip = r < flip_prob
    image, labels = tf.cond( to_flip, lambda: flip(image,labels), lambda: identity(image,labels) )

    return image, labels

def process_image_and_labels(img_path, lbl_path, image_format, image_means, target_height, target_width):
    """
    Load, scale and optionally augment images and labels
    :param img_path: complete file name of image
    :param lbl_path: complete name of label file
    :return: image, [scale_x, scale_y], labels
    """
    image, [scale_x, scale_y] = process_image(img_path, image_format, image_means, target_height, target_width)

    # Note that the function 'json2boxes' returns a single np multi-dimensional array but it retuturns a list
    # In our case the list has a signle element
    labels = tf.py_func(json2bboxes, [lbl_path], [tf.float64],  name='json2bboxes')[0]
    scale = tf.cast(tf.stack([scale_x, scale_y, scale_x, scale_y]), labels.dtype)
    labels = labels * scale

    if self.data_augmentation:
        image, labels = self.random_flip(image,labels)

    return image, [scale_x, scale_y], labels

def json2bboxes(json_fname):
    df = NexarLabelsBase.json_to_df(json_fname)

    label_objects = []
    grouped = df.groupby([TYPE, TYPE_REPRESENTATION])
    for name, group in grouped:
        label_type = name[0]
        if label_type == 'NexodBox':
            box_obj = name[1]
            label_objects.append(box_obj)

    if len(label_objects) > 0:
        bb = np.array([(b.xmin(),b.ymin(),b.xmax(),b.ymax())  for b in  label_objects])
    else:
        bb = np.array([(-1.,-1.,-1.,-1.)])

    return bb
