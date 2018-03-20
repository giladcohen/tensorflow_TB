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
from object_detection.data_decoders import tf_example_decoder
from object_detection.core import standard_fields as fields
from object_detection.core import batcher

def _get_padding_shapes(dataset, max_num_boxes, num_classes, spatial_image_shape):
  """Returns shapes to pad dataset tensors to before batching.

  Args:
    dataset: tf.data.Dataset object.
    max_num_boxes: Max number of groundtruth boxes needed to computes shapes for
      padding.
    num_classes: Number of classes in the dataset needed to compute shapes for
      padding.
    spatial_image_shape: A list of two integers of the form [height, width]
      containing expected spatial shape of the imaage.

  Returns:
    A dictionary keyed by fields.InputDataFields containing padding shapes for
    tensors in the dataset.
  """
  height, width = spatial_image_shape
  padding_shapes = {
      fields.InputDataFields.image: [height, width, 3],
      fields.InputDataFields.source_id: [],
      fields.InputDataFields.filename: [],
      fields.InputDataFields.key: [],
      fields.InputDataFields.groundtruth_difficult: [max_num_boxes],
      fields.InputDataFields.groundtruth_boxes: [max_num_boxes, 4],
      fields.InputDataFields.groundtruth_classes: [max_num_boxes, num_classes],
      fields.InputDataFields.groundtruth_instance_masks: [max_num_boxes, height, width],
      fields.InputDataFields.groundtruth_is_crowd: [max_num_boxes],
      fields.InputDataFields.groundtruth_group_of: [max_num_boxes],
      fields.InputDataFields.groundtruth_area: [max_num_boxes],
      fields.InputDataFields.groundtruth_weights: [max_num_boxes],
      fields.InputDataFields.num_groundtruth_boxes: [],
      fields.InputDataFields.groundtruth_label_types: [max_num_boxes],
      fields.InputDataFields.groundtruth_label_scores: [max_num_boxes],
      fields.InputDataFields.true_image_shape: [3]
  }
  if fields.InputDataFields.groundtruth_keypoints in dataset.output_shapes:
    tensor_shape = dataset.output_shapes[fields.InputDataFields.groundtruth_keypoints]
    padding_shape = [max_num_boxes, tensor_shape[1].value, tensor_shape[2].value]
    padding_shapes[fields.InputDataFields.groundtruth_keypoints] = padding_shape
  if (fields.InputDataFields.groundtruth_keypoint_visibilities in dataset.output_shapes):
    tensor_shape = dataset.output_shapes[fields.InputDataFields.groundtruth_keypoint_visibilities]
    padding_shape = [max_num_boxes, tensor_shape[1].value]
    padding_shapes[fields.InputDataFields.groundtruth_keypoint_visibilities] = padding_shape
  return {tensor_key: padding_shapes[tensor_key] for tensor_key, _ in dataset.output_shapes.items()}

def create_input_queue(batch_size, tensor_dict):

  tensor_dict[fields.InputDataFields.image] = tf.expand_dims(
      tensor_dict[fields.InputDataFields.image], 0)

  images = tensor_dict[fields.InputDataFields.image]
  float_images = tf.to_float(images)
  tensor_dict[fields.InputDataFields.image] = float_images

  input_queue = batcher.BatchQueue(
      tensor_dict,
      batch_size=batch_size,
      batch_queue_capacity=3*batch_size,
      num_batch_queue_threads=5,
      prefetch_queue_capacity=3*batch_size)
  return input_queue

class NexetDatasetWrapper(DatasetWrapper):
    def __init__(self, *args, **kwargs):
        super(NexetDatasetWrapper, self).__init__(*args, **kwargs)

        self.dataset_path         = self.prm.dataset.DATASET_PATH
        self.train_dir            = os.path.join(self.dataset_path, 'train')
        self.test_dir             = os.path.join(self.dataset_path, 'eval')
        self.label_map_proto_file = os.path.join(self.dataset_path, 'NEXET_label_map.pbtxt')
        self.train_input_queue    = None

    def build_datasets(self):
        """Building the NEXET dataset"""
        self.train_dataset      = self.set_transform('train'     , Mode.TRAIN)
        self.test_dataset       = self.set_transform('test'      , Mode.EVAL)

    def set_transform(self, name, mode, batch_size=None, **kwargs):
        decoder = tf_example_decoder.TfExampleDecoder(label_map_proto_file=self.label_map_proto_file)
        if name in ['train', 'train_eval']:
            filename = os.path.join(self.train_dir, 'NEXET_train_1_of_1.tfrecord')
        else:
            filename = os.path.join(self.test_dir , 'NEXET_eval_1_of_1.tfrecord')
        if batch_size is None:
            if mode == Mode.TRAIN:
                batch_size = self.prm.train.train_control.TRAIN_BATCH_SIZE
            else:
                batch_size = self.prm.train.train_control.EVAL_BATCH_SIZE

        def _decode_func(value):
            processed = decoder.decode(value)
            # if transform_input_data_fn is not None:
            #     return transform_input_data_fn(processed)
            return processed

        def _extract_images_and_targets(read_data):
            label_id_offset = 1
            image = read_data[fields.InputDataFields.image]
            # image = tf.expand_dims(image, 0)
            image = tf.to_float(image)

            key = ''
            if fields.InputDataFields.source_id in read_data:
                key = read_data[fields.InputDataFields.source_id]
            location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
            classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes], tf.int32)
            classes_gt -= label_id_offset
            classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt, depth=4, left_pad=0)  # num_classes=4
            # masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
            # keypoints_gt = read_data.get(fields.InputDataFields.groundtruth_keypoints)
            # weights_gt = read_data.get(fields.InputDataFields.groundtruth_weights)

            # not in use by trainer, only by evaluator
            # is_crowd_gt = read_data[fields.InputDataFields.groundtruth_is_crowd]
            # area_gt = read_data[fields.InputDataFields.groundtruth_area]
            # group_of_gt = read_data[fields.InputDataFields.groundtruth_group_of]
            # difficult_gt = read_data[fields.InputDataFields.groundtruth_difficult]
            # source_id = read_data[fields.InputDataFields.source_id]

            # return image, key, location_gt, classes_gt, weights_gt, is_crowd_gt, area_gt, group_of_gt, difficult_gt, source_id
            return image, location_gt, classes_gt

        with tf.name_scope(name + '_data'):
            # feed all datasets with the same model placeholders:
            dataset = tf.data.Dataset.from_tensor_slices([filename])
            dataset = dataset.repeat()
            dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=1, block_length=1)
            dataset = dataset.shuffle(
                    buffer_size=batch_size,
                    seed=self.prm.SUPERSEED,
                    reshuffle_each_iteration=True)
            dataset = dataset.map(_decode_func, num_parallel_calls=batch_size)
            dataset = dataset.prefetch(2 * batch_size)

            max_num_boxes = 100
            spatial_image_shape = [272, 480]
            padding_shapes = _get_padding_shapes(dataset, max_num_boxes, self.num_classes, spatial_image_shape)
            dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padding_shapes))

            # if mode == Mode.TRAIN:
            #     max_num_boxes = 50
            #     spatial_image_shape = [272, 480]
            #     padding_shapes = _get_padding_shapes(dataset, max_num_boxes, self.num_classes, spatial_image_shape)
            #     dataset = dataset.apply(tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padding_shapes))

            #     # dataset = dataset.map(map_func=_augment, num_parallel_calls=batch_size)
            #     dataset = dataset.shuffle(
            #         buffer_size=batch_size,
            #         seed=self.prm.SUPERSEED,
            #         reshuffle_each_iteration=True)
            #     dataset = dataset.prefetch(2 * batch_size)
            #     dataset = dataset.repeat()
            #     dataset = dataset.map(_extract_images_and_targets, num_parallel_calls=batch_size)
            #     dataset = dataset.batch(batch_size)
            # else:
            #     dataset = dataset.map(_extract_images_and_targets, num_parallel_calls=batch_size)
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
        # self.handle = tf.placeholder(tf.string, shape=[])
        # self.iterator = tf.data.Iterator.from_string_handle(
        #     self.handle, self.train_dataset.output_types, self.train_dataset.output_shapes)
        # self.next_minibatch = self.iterator.get_next()

        # generate iterators
        # self.train_iterator      = self.train_dataset.make_one_shot_iterator()
        self.train_iterator        = self.train_dataset.make_initializable_iterator()
        self.test_iterator         = self.test_dataset.make_initializable_iterator()

        self.next_minibatch_train  = self.train_iterator.get_next()
        self.next_minibatch_test   = self.test_iterator.get_next()

        self.train_input_queue     = create_input_queue(self.train_batch_size, self.next_minibatch_train)

        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, self.train_iterator.initializer)
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, self.test_iterator.initializer)
