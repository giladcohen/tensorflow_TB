from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from utils.enums import Mode
from lib.datasets.dataset_wrapper import DatasetWrapper
from utils.misc import get_full_names_of_image_files
import json
import numpy as np
import tensorflow as tf
from object_detection.utils import ops as util_ops

class NexetDatasetWrapper(DatasetWrapper):

    def __init__(self, *args, **kwargs):
        super(NexetDatasetWrapper, self).__init__(*args, **kwargs)

        self.map_class_names = {"car": 1,
                                "pickup_truck": 2,
                                "truck": 3,
                                "bus": 4,
                                "van": 5,
                                "motorcycle": 6,
                                "bicycle": 7}

        self.dataset_path         = self.prm.dataset.DATASET_PATH
        self.train_dir            = os.path.join(self.dataset_path, 'train')
        self.train_raw_images_dir = os.path.join(self.train_dir, 'images')
        self.train_images_file    = os.path.join(self.train_dir, 'images.txt')
        self.train_labels_dir     = os.path.join(self.train_dir, 'annotations', 'labels')
        self.test_dir             = os.path.join(self.dataset_path, 'test')
        self.test_raw_images_dir  = os.path.join(self.test_dir, 'images')
        self.test_images_file     = os.path.join(self.test_dir, 'images.txt')
        self.test_labels_dir      = os.path.join(self.test_dir, 'annotations', 'labels')

        self.max_num_boxes = 17

    def get_labels(self, labels_dir, base_image_fnames):
        max_num_boxes = 0
        set_size    = len(base_image_fnames)
        bbox_labels = -1 * np.ones([set_size, self.max_num_boxes, 4], dtype=np.float64)
        cls_labels  = -1 * np.ones([set_size, self.max_num_boxes], dtype=np.int32)
        num_boxes   = np.zeros(shape=[set_size], dtype=np.int32)

        for i, base_image_fname in enumerate(base_image_fnames):
            json_path = os.path.join(labels_dir, base_image_fname + '.json')
            with open(json_path) as infile:
                nexar_labels_dict = json.load(infile)
            for j, single_label in enumerate(nexar_labels_dict):
                num_boxes[i] += 1
                x0 = single_label['type_representation']['x0']  # / 720.0
                y0 = single_label['type_representation']['y0']  # / 1280.0
                x1 = single_label['type_representation']['x1']  # / 720.0
                y1 = single_label['type_representation']['y1']  # / 1280.0
                bbox_labels[i, j] = np.array([x0, y0, x1, y1], dtype=np.float32)
                cls_str = single_label['class_name']['objects_on_the_road_01']['tag']
                cls_int = self.map_class_names[cls_str]
                cls_labels[i, j] = cls_int

            if num_boxes[i] > max_num_boxes:
                max_num_boxes = num_boxes[i]
        print('max_num_boxes={}'.format(max_num_boxes))
        return [bbox_labels, cls_labels, num_boxes]

    def build_datasets(self):
        """Building the NEXET dataset"""

        # building the train set
        image_fnames = get_full_names_of_image_files(self.train_raw_images_dir, self.train_images_file)
        image_fnames.sort()
        base_image_fnames = [os.path.basename(im) for im in image_fnames]
        labels = self.get_labels(self.train_labels_dir, base_image_fnames)
        self.train_dataset = self.set_transform('train', Mode.TRAIN, image_fnames, labels)

        # building the test set
        image_fnames = get_full_names_of_image_files(self.test_raw_images_dir, self.test_images_file)
        image_fnames.sort()
        base_image_fnames = [os.path.basename(im) for im in image_fnames]
        labels = self.get_labels(self.test_labels_dir, base_image_fnames)
        self.test_dataset = self.set_transform('test', Mode.EVAL, image_fnames, labels)

    def set_transform(self, name, mode, image_fnames, labels, **kwargs):
        """
        :param **kwargs: **kwargs
        :param name: name
        :param mode: mode
        :param image_fnames: image_fnames
        :param labels: labels
        :return: tf.data.Dataset
        """

        def _map_to_image_label(image_fname, bbox_label, cls_label, num_boxes):
            # read the img from file
            img_file = tf.read_file(image_fname)
            image = tf.image.decode_image(img_file, channels=3)
            image.set_shape([None, None, 3])

            # process image - resize
            source_h = tf.shape(image)[0]
            source_w = tf.shape(image)[1]

            image = tf.image.resize_images(image, [272, 480])
            scale_y = tf.to_double(tf.shape(image)[0] / source_h)
            scale_x = tf.to_double(tf.shape(image)[1] / source_w)

            # process image - normalize
            # image = tf.image.per_image_standardization(image)
            norm  = tf.cast(tf.stack([source_w, source_h, source_w, source_h]), tf.float64)

            # bbox_label = tf.cast(bbox_label, tf.float64)
            # num_boxes = tf.expand_dims(num_boxes, -1)

            bbox_label = bbox_label / norm
            return image_fname, image, bbox_label, cls_label, num_boxes

        # def _reshape_label(image_fname, image, bbox_label, cls_label, num_boxes):
        #     """
        #     Convert cls_label to list of one-hot vectors
        #     """
        #     cls_label = util_ops.padded_one_hot_encoding(indices=cls_label, depth=self.num_classes, left_pad=0)
        #     return image_fname, image, cls_label, bbox_label, num_boxes

        if mode == Mode.TRAIN:
            batch_size = self.prm.train.train_control.TRAIN_BATCH_SIZE
        else:
            batch_size = self.prm.train.train_control.EVAL_BATCH_SIZE

        bboxes, classes, num_boxes = labels
        with tf.name_scope(name + '_data'):
            dataset = tf.data.Dataset.from_tensor_slices((image_fnames, bboxes, classes, num_boxes))
            dataset = dataset.map(map_func=_map_to_image_label, num_parallel_calls=batch_size)
            # dataset = dataset.map(map_func=_reshape_label, num_parallel_calls=batch_size)

            if mode == Mode.TRAIN:
                # dataset = dataset.map(map_func=_augment, num_parallel_calls=batch_size)
                dataset = dataset.shuffle(
                    buffer_size=batch_size,
                    seed=self.prm.SUPERSEED,
                    reshuffle_each_iteration=True)
                dataset = dataset.prefetch(2 * batch_size)
                dataset = dataset.repeat()
            else:
                dataset = dataset.prefetch(2 * batch_size)

            dataset = dataset.batch(batch_size)
            return dataset

    def build_iterators(self):
        """
        Sets the train/validation/test/train_eval iterators
        :return: None
        """
        # A feedable iterator is defined by a handle placeholder and its structure. We
        # could use the `output_types` and `output_shapes` properties of either
        # `training_dataset` or `validation_dataset` here, because they have
        # identical structure.
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(
            self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
        self.next_minibatch = self.iterator.get_next()

        # generate iterators
        self.train_iterator      = self.train_dataset.make_one_shot_iterator()
        self.test_iterator       = self.test_dataset.make_initializable_iterator()

    def set_handles(self, sess):
        """
        set the handles. Must be called from the trainer/tester, using a session
        :param sess: session
        :return: None
        """
        # The `Iterator.string_handle()` method returns a tensor that can be evaluated
        # and used to feed the `handle` placeholder.
        self.train_handle      = sess.run(self.train_iterator.string_handle())
        self.test_handle       = sess.run(self.test_iterator.string_handle())

    def get_handle(self, name):
        """Getting an iterator handle based on dataset name
        :param name: name of the dataset (string). e.g., 'train', 'train_eval', 'validation', 'test', etc.
        """
        if name == 'train':
            return self.train_handle
        elif name == 'test':
            return self.test_handle

        err_str = 'calling get_mini_batch with illegal dataset name ({})'.format(name)
        self.log.error(err_str)
        raise AssertionError(err_str)

    def get_mini_batch(self, name, sess):
        """
        Get a session and returns the next training batch
        :param name: the name of the dataset
        :param sess: Session
        :return: next training batch
        """
        handle = self.get_handle(name)
        image_fnames, images, bbox_labels, cls_labels, num_boxes = sess.run(self.next_minibatch, feed_dict={self.handle: handle})
        return image_fnames, images, bbox_labels, cls_labels, num_boxes
