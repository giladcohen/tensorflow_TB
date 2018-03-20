# --------------------------------------------------------------------------------------------------
#
#   Copyright (c) 2016-2017. Nexar Inc. - All Rights Reserved. Proprietary and confidential.
#
#   Unauthorized copying of this file, via any medium is strictly prohibited.
#
# --------------------------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.contrib.data import Dataset

from src.datasets.preprocessors.preprocess_funcs import process_image_and_labels, process_image
from g_model.base.utils import parse_images_params_str, get_full_names_of_image_files

from src.datasets.dataset_hooks import IteratorInitializerHook


def get_nexar_train_eval_inputs(params):
    """Return the input function to get the training data.

    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.

    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    train_iterator_initializer_hook = IteratorInitializerHook()
    test_iterator_initializer_hook = IteratorInitializerHook()

    target_height = params.target_height
    target_width = params.target_width

    try:
        images_dir, images_file = parse_images_params_str(params.IMAGES_LOCATION)
        images_dir = os.path.join(images_dir, '')  # Add trailing separator if not already there
    except:
        err_str = 'Failed to parse image directory from parameter: {}'.format(params.IMAGES_LOCATION)
        raise ValueError(err_str)
    labels_dir = os.path.join(params.LABELS_DIR, '')  # Add trailing separator if not already there

    def map_train_func(base_image_fname):
        """ Should be tensorflow function"""
        # read the img from file
        img_path = images_dir + base_image_fname
        json_path = labels_dir + base_image_fname + ".json"
        images, scales, labels = process_image_and_labels(img_path, json_path,
                                                          params.image_format, params.image_means,
                                                          params.target_height, params.target_width)
        return [base_image_fname], images, scales, labels

    def map_test_func(base_image_fname):
        """ Should be tensorflow function"""
        # read the img from file
        img_path = images_dir + base_image_fname
        images, scales = process_image(img_path,
                                       params.image_format, params.image_means,
                                       params.target_height, params.target_width)
        return [base_image_fname], images, scales

    def train_inputs():
        """Returns training set as Operations.

        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Training_data'):
            images_shape = [target_height, target_width, 3]
            bboxes_shape = [None, 4]  # Sometimes there are no boxes for an image
            img_name_shape = [1]
            image_scales_shape = [2]

            image_fnames = get_full_names_of_image_files(images_dir, images_file)
            image_fnames.sort()
            base_image_fnames = [os.path.basename(im) for im in image_fnames]

            dataset = Dataset.from_tensor_slices(base_image_fnames)
            dataset = dataset.take(params.NUMBER_ELEMENTS_TO_TAKE)
            dataset = dataset.map(map_train_func)
            if params.MINI_BATCH_SHUFFLING_SEED > 0:
                dataset = dataset.shuffle(buffer_size=len(base_image_fnames),
                                                    seed=params.MINI_BATCH_SHUFFLING_SEED)
            dataset = dataset.repeat(params.REPEAT)
            dataset = dataset.padded_batch(params.BATCH_SIZE,
                                           (img_name_shape, images_shape, image_scales_shape, bboxes_shape),
                                           ("unknown",
                                              tf.convert_to_tensor(-1.0, dtype=tf.float32),
                                              tf.convert_to_tensor(-1.0, dtype=tf.float64),
                                              tf.convert_to_tensor(-1.0, dtype=tf.float64)))

            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set runhook to initialize iterator
            train_iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)
            # Return batched (features, labels)
            return next_example, next_label

    def test_inputs():
        """Returns training set as Operations.

        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Test_data'):
            images_shape = [target_height, target_width, 3]
            bboxes_shape = [None, 4]  # Sometimes there are no boxes for an image
            img_name_shape = [1]
            image_scales_shape = [2]

            image_fnames = get_full_names_of_image_files(images_dir, images_file)
            image_fnames.sort()
            base_image_fnames = [os.path.basename(im) for im in image_fnames]

            dataset = Dataset.from_tensor_slices(base_image_fnames)
            dataset = dataset.take(params.NUMBER_ELEMENTS_TO_TAKE)
            dataset = dataset.map(map_train_func)
            if params.MINI_BATCH_SHUFFLING_SEED > 0:
                dataset = dataset.shuffle(buffer_size=len(base_image_fnames),
                                                    seed=params.MINI_BATCH_SHUFFLING_SEED)
            dataset = dataset.repeat(params.REPEAT)
            dataset = dataset.padded_batch(params.BATCH_SIZE,
                                          (img_name_shape, images_shape, image_scales_shape))
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set runhook to initialize iterator
            train_iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)
            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return {'train_inputs' : train_inputs,
            'train_iterator_initializer_hook' : train_iterator_initializer_hook,
            'test_inputs': test_inputs,
            'test_iterator_initializer_hook': test_iterator_initializer_hook
            }


# class NexarDataset(AgentBase):
#     def __init__(self, name, prm, preprocessor):
#         super(NexarDataset, self).__init__(name)
#         self.prm = prm
#
#         try:
#             self.images_dir, self.images_file = parse_images_params_str(self.prm.IMAGES_LOCATION)
#             self.images_dir = os.path.join(self.images_dir,'') # Add trailing separator if not already there
#         except:
#             err_str = 'Failed to parse image directory from parameter: {}'.format(self.prm.IMAGES_LOCATION)
#             self.log.exception(err_str)
#             raise Exception(err_str)
#
#         self.labels_dir = os.path.join(prm.LABELS_DIR,'') # Add trailing separator if not already there
#         self.preprocessor = preprocessor
#         target_height = preprocessor.target_height
#         target_width = preprocessor.target_width
#         images_shape = [target_height, target_width, 3]
#         bboxes_shape = [None, 4]     # Sometimes there are no boxes for an image
#         img_name_shape = [1]
#         image_scales_shape = [2]
#
#         self.image_fnames = get_full_names_of_image_files(self.images_dir, self.images_file)
#         self.image_fnames.sort()
#         self.base_image_fnames = [os.path.basename(im) for im in self.image_fnames]
#
#         self.dataset = Dataset.from_tensor_slices(self.base_image_fnames)
#         self.dataset = self.dataset.take(self.prm.NUMBER_ELEMENTS_TO_TAKE)
#         self.dataset = self.dataset.map(self.map_func)
#         if self.prm.MINI_BATCH_SHUFFLING_SEED > 0:
#             self.dataset = self.dataset.shuffle(buffer_size=len(self.base_image_fnames), seed=self.prm.MINI_BATCH_SHUFFLING_SEED)
#         self.dataset = self.dataset.repeat(self.prm.REPEAT)
#         if self.labels_dir is None:
#             self.dataset = self.dataset.padded_batch(self.prm.BATCH_SIZE,
#                                                      (img_name_shape, images_shape, image_scales_shape))
#         else:
#             self.dataset = self.dataset.padded_batch(self.prm.BATCH_SIZE,
#                                                      (img_name_shape, images_shape, image_scales_shape, bboxes_shape),
#                                                      ("unknown",
#                                                       tf.convert_to_tensor(-1.0,dtype=tf.float32),
#                                                       tf.convert_to_tensor(-1.0,dtype=tf.float64),
#                                                       tf.convert_to_tensor(-1.0, dtype=tf.float64)))
#
#
#
#     def __str__(self):
#         str = 'DATASET_NAME:{}\n  BATCH_SIZE:{}'.format(self.name, self.prm.BATCH_SIZE)
#         return str
#
#     def print_stats(self):
#         self.log.info('NexarDataset:   {}'.format(self.name))
#         self.log.info(' IMAGE DIR:    {}'.format(self.get_images_dir()))
#         self.log.info(' LABELS DIR:    {}'.format(self.get_labels_dir()))
#         self.log.info(' TOTAL SIZE:    {}'.format(self.get_number_of_images()))
#         self.log.info(' RESTRICTED TO: {}'.format(self.prm.NUMBER_ELEMENTS_TO_TAKE))
#         self.log.info(' REPEATED:      {}'.format(self.prm.REPEAT))
#         self.log.info(' BATCH_SIZE:    {}'.format(self.prm.BATCH_SIZE))
#         self.log.info(' MINI_BATCH_SHUFFLING_SEED: {}'.format(self.prm.MINI_BATCH_SHUFFLING_SEED))
#         self.preprocessor.print_stats()
#
#     def make_one_shot_iterator(self):
#         return self.dataset.make_one_shot_iterator()
#
#     def get_images_dir(self):
#         return self.images_dir
#
#     def get_labels_dir(self):
#         return self.labels_dir
#
#     def get_number_of_images(self):
#         return len(self.base_image_fnames)
#
#     def get_batch_size(self):
#         return self.prm.BATCH_SIZE
#
#     def map_func(self, base_image_fname):
#         """ Should be tensorflow function"""
#         # read the img from file
#         img_path = self.images_dir + base_image_fname
#         if self.labels_dir is None:
#             images, scales = self.preprocessor.process_image(img_path)
#             return [base_image_fname], images, scales
#         else:
#             json_path = self.labels_dir + base_image_fname + ".json"
#             images, scales, labels = self.preprocessor.process_image_and_labels(img_path, json_path)
#             return [base_image_fname], images, scales, labels


