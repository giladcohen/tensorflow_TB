from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
from sklearn import preprocessing
from lib.datasets.dataset_wrapper import DatasetWrapper
from utils.enums import Mode
import tensorflow as tf
import json
from utils.misc import get_full_names_of_image_files
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

        self.dataset_path     = self.prm.dataset.DATASET_PATH
        self.train_dir        = os.path.join(self.dataset_path, 'train')
        self.train_images_dir = os.path.join(self.train_dir, 'images')
        self.train_labels_dir = os.path.join(self.train_dir, 'annotations', 'labels')
        self.test_dir         = os.path.join(self.dataset_path, 'test')
        self.test_images_dir  = os.path.join(self.test_dir, 'images')
        self.test_labels_dir  = os.path.join(self.test_dir, 'annotations', 'labels')

        self.train_labels     = None
        self.test_labels      = None

    def get_labels(self, labels_dir, base_image_fnames):
        labels = []
        for base_image_fname in base_image_fnames:
            json_path = os.path.join(labels_dir, base_image_fname + '.json')
            row = {}
            row["classes"]      = []
            row["bboxes"]       = []
            with open(json_path) as infile:
                nexar_labels_dict = json.load(infile)
            for single_label in nexar_labels_dict:
                cls_str = single_label['class_name']['objects_on_the_road_01']['tag']
                cls_int = self.map_class_names[cls_str]
                row["classes"].append(cls_int)
                x0 = single_label['type_representation']['x0']
                y0 = single_label['type_representation']['y0']
                x1 = single_label['type_representation']['x1']
                y1 = single_label['type_representation']['y1']
                row["bboxes"].append(np.array([x0, y0, x1, y1], dtype=np.float32))
                labels.append(row)
        return labels

    def build_datasets(self):
        """Building the NEXET dataset"""
        images_file = os.path.join(self.train_dir, 'images.txt')
        image_fnames = get_full_names_of_image_files(self.train_images_dir, images_file)
        image_fnames.sort()
        base_image_fnames = [os.path.basename(im) for im in image_fnames]
        self.train_labels = self.get_labels(self.train_labels_dir, base_image_fnames)
        self.train_dataset      = self.set_transform('train'     , Mode.TRAIN, image_fnames, self.train_labels)
        self.train_eval_dataset = self.set_transform('train_eval', Mode.EVAL , image_fnames, self.train_labels)

        images_file = os.path.join(self.test_dir, 'images.txt')
        image_fnames = get_full_names_of_image_files(self.test_images_dir, images_file)
        image_fnames.sort()
        base_image_fnames = [os.path.basename(im) for im in image_fnames]
        self.test_labels = self.get_labels(self.test_labels_dir, base_image_fnames)
        self.test_dataset = self.set_transform('test', Mode.EVAL, image_fnames, self.test_labels)

    def set_transform(self, name, mode, image_fnames, labels, **kwargs):
        """
        :param **kwargs: **kwargs
        :param name: name
        :param mode: mode
        :param image_fnames: image_fnames
        :param labels: labels
        :return: tf.data.Dataset
        """

        def _identity(image, bbox_label):
            return image, bbox_label

        def _flip(image, bbox_label):
            """Flipping"""
            image = tf.image.flip_left_right(image)
            # wx = tf.to_double(tf.shape(image)[1])
            for i in range(len(bbox_label)):
                bbox_label[i] = tf.stack([1.0 - bbox_label[i][0],
                                                bbox_label[i][1],
                                          1.0 - bbox_label[i][2],
                                                bbox_label[i][3]])
            return image, bbox_label

        def _random_flip(image, bbox_label):
            r = tf.random_uniform(shape=())
            to_flip = r < 0.5
            image, bbox_label = tf.cond(to_flip, lambda: _flip(image, bbox_label), lambda: _identity(image, bbox_label))
            return image, bbox_label

        def _map_to_image_label(image_fname, cls_label, bbox_label):
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
            image = tf.image.per_image_standardization(image)
            norm  = tf.cast(tf.stack([source_w, source_h, source_w, source_h]), tf.float32)

            bbox_label = bbox_label / norm

            return image_fname, image, [scale_x, scale_y], cls_label, bbox_label

        # TODO(gilad): add ssd_random_crop
        def _augment(image_fname, image, scale, cls_label, bbox_label):
            """Augmenting each image/label in the dataset"""

            # flip
            image, bbox_label = _random_flip(image, bbox_label)
            return image_fname, image, scale, cls_label, bbox_label

        def _reshape_label(image_fname, image, scale, cls_label, bbox_label):
            """Reshaping to a list of labels with shape=[NUM_BBOXES_IN_IMAGE, 4]
            Convert label["bboxes"] from list of NUM_BOXES elements of 4 values to [NUM_BOXES, 4] shape.
            Convert label["classes] for list of values to list of one-hot vectors
            """
            # cls_label = tf.expand_dims(cls_label, 1)
            cls_label = tf.map_fn(
                lambda cls: util_ops.padded_one_hot_encoding(cls, depth=self.num_classes, left_pad=0),
                cls_label,
                dtype=tf.int32)
            cls_label  = tf.stack(cls_label)
            # bbox_label = tf.stack(bbox_label)
            return image_fname, image, scale, cls_label, bbox_label

        if mode == Mode.TRAIN:
            batch_size = self.prm.train.train_control.TRAIN_BATCH_SIZE
        else:
            batch_size = self.prm.train.train_control.EVAL_BATCH_SIZE

        with tf.name_scope(name + '_data'):
            # feed all datasets with the same model placeholders:
            cls_labels_list  = []
            bbox_labels_list = []
            for l in labels:
                cls_labels_list.append(np.array(l['classes']))
                bbox_labels_list.append((np.array(l['bboxes'])))

            dataset = tf.data.Dataset.from_tensor_slices((image_fnames, cls_labels_list, bbox_labels_list))
            dataset = dataset.map(map_func=_map_to_image_label, num_parallel_calls=batch_size)

            if mode == Mode.TRAIN:
                dataset = dataset.map(map_func=_augment, num_parallel_calls=batch_size)
                dataset = dataset.shuffle(
                    buffer_size=batch_size,
                    seed=self.prm.SUPERSEED,
                    reshuffle_each_iteration=True)
                dataset = dataset.prefetch(2 * batch_size)
                dataset = dataset.repeat()

            dataset = dataset.map(map_func=_reshape_label, num_parallel_calls=batch_size)
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
        self.train_eval_iterator = self.train_eval_dataset.make_initializable_iterator()
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
        self.train_eval_handle = sess.run(self.train_eval_iterator.string_handle())
        self.test_handle       = sess.run(self.test_iterator.string_handle())

    def get_handle(self, name):
        """Getting an iterator handle based on dataset name
        :param name: name of the dataset (string). e.g., 'train', 'train_eval', 'validation', 'test', etc.
        """
        if name == 'train':
            return self.train_handle
        elif name == 'train_eval':
            return self.train_eval_handle
        elif name == 'test':
            return self.test_handle

        err_str = 'calling get_mini_batch with illegal dataset name ({})'.format(name)
        self.log.error(err_str)
        raise AssertionError(err_str)
