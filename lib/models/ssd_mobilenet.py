from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from lib.models.model_base import ModelBase
import tensorflow as tf
slim = tf.contrib.slim
# from nets import mobilenet_v1
# from object_detection.models import feature_map_generators
import functools
from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
from object_detection.box_coders import faster_rcnn_box_coder
from object_detection.matchers import argmax_matcher
from object_detection.core.region_similarity_calculator import IouSimilarity
from object_detection.core.box_predictor import ConvolutionalBoxPredictor
from object_detection.anchor_generators import multiple_grid_anchor_generator
from object_detection.core.preprocessor import resize_image
from object_detection.core.post_processing import batch_multiclass_non_max_suppression
from object_detection.builders.post_processing_builder import _score_converter_fn_with_logit_scale
from object_detection.core.losses import WeightedSigmoidClassificationLoss, WeightedSmoothL1LocalizationLoss, HardExampleMiner
# from object_detection.meta_architectures.ssd_meta_arch import SSDMetaArch
from lib.models.ssd_meta_arch_v2 import SSDMetaArch_V2
from lib.base.collections import LOSSES
from object_detection.core import standard_fields as fields
from object_detection import eval_util
from object_detection.utils import ops as util_ops

class SSDMobileNet(ModelBase):

    def __init__(self, *args, **kwargs):
        super(SSDMobileNet, self).__init__(*args, **kwargs)
        self.num_classes        = self.prm.network.NUM_CLASSES
        self.ssd_meta_arch      = None
        self.train_batch_size   = self.prm.train.train_control.TRAIN_BATCH_SIZE
        self.eval_batch_size    = self.prm.train.train_control.EVAL_BATCH_SIZE
        self.ignore_groundtruth = False

    def print_stats(self):
        super(SSDMobileNet, self).print_stats()
        self.log.info(' NUM_CLASSES: {}'.format(self.num_classes))

    def _build_interpretation(self):
        images      = self.eval_images
        bbox_labels = self.eval_bbox_labels
        cls_labels  = self.eval_cls_labels
        num_boxes   = self.eval_num_boxes
        source_ids  = self.eval_source_ids

        images.set_shape([self.eval_batch_size, 272, 480, 3])
        bbox_labels.set_shape([self.eval_batch_size, None, 4])
        cls_labels.set_shape([self.eval_batch_size, None])
        num_boxes.set_shape([self.eval_batch_size])
        source_ids.set_shape([self.eval_batch_size])

        bbox_labels = tf.squeeze(bbox_labels, axis=0)
        cls_labels  = tf.squeeze(cls_labels, axis=0)
        num_boxes   = tf.squeeze(num_boxes, axis=0)
        source_ids  = tf.squeeze(source_ids, axis=0)

        groundtruth_boxes   = tf.slice(bbox_labels, [0, 0], [num_boxes, 4], name='eval_boxes_tensor_slice')
        groundtruth_classes = tf.slice(cls_labels, [0], [num_boxes], name='eval_cls_tensor_slice')

        resized_images, true_images_shape = self.ssd_meta_arch.preprocess(tf.to_float(images))
        prediction_dict = self.ssd_meta_arch.predict(resized_images, true_images_shape)
        detections = self.ssd_meta_arch.postprocess(prediction_dict, true_images_shape)

        groundtruth = None
        if not self.ignore_groundtruth:
            groundtruth = {
                fields.InputDataFields.groundtruth_boxes:     groundtruth_boxes,
                fields.InputDataFields.groundtruth_classes:   groundtruth_classes
                # fields.InputDataFields.groundtruth_area:      None,
                # fields.InputDataFields.groundtruth_is_crowd:  None,
                # fields.InputDataFields.groundtruth_difficult: None
            }
            # if fields.InputDataFields.groundtruth_group_of in input_dict:
            #     groundtruth[fields.InputDataFields.groundtruth_group_of] = (
            #         input_dict[fields.InputDataFields.groundtruth_group_of])
            # if fields.DetectionResultFields.detection_masks in detections:
            #     groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
            #         input_dict[fields.InputDataFields.groundtruth_instance_masks])

        self.tensor_dict = eval_util.result_dict_for_single_example(
            images,
            source_ids,
            detections,
            groundtruth,
            class_agnostic=(fields.DetectionResultFields.detection_classes not in detections),
            scale_to_absolute=True)

    def _init_params(self):
        super(SSDMobileNet, self)._init_params()
        self.iou_rate = tf.contrib.framework.model_variable(
            name='iou_rate', dtype=tf.float32, shape=[],
            initializer=tf.constant_initializer(self.prm.network.optimization.IOU_RATE), trainable=False)

    def _set_params(self):
        super(SSDMobileNet, self)._set_params()
        self.assign_ops['iou_rate'] = self.iou_rate.assign(self.prm.network.optimization.IOU_RATE)

    def _build_inference(self):
        """Building the mobile net"""

        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(scale=float(0.00004)),
            weights_initializer=tf.truncated_normal_initializer(mean=0.03, stddev=0.0),
            activation_fn=tf.nn.relu6,
            normalizer_fn=slim.batch_norm,
            normalizer_params={
                'decay': 0.9997,
                'center': True,
                'scale': True,
                'epsilon': 0.001,
                'is_training': self.is_training}) as sc:
            feature_extractor = SSDMobileNetV1FeatureExtractor(
                is_training=self.is_training,
                depth_multiplier=1.0,
                min_depth=16,
                pad_to_multiple=1,
                conv_hyperparams=sc,
                reuse_weights=None)

        y_scale      = 10.0
        x_scale      = 10.0
        height_scale = 5.0
        width_scale  = 5.0

        box_coder = faster_rcnn_box_coder.FasterRcnnBoxCoder(scale_factors=[y_scale, x_scale, height_scale,width_scale])

        matcher = argmax_matcher.ArgMaxMatcher(
            matched_threshold=0.5,
            unmatched_threshold=0.5,
            negatives_lower_than_unmatched=True,
            force_match_for_each_row=True)

        region_similarity_calculator = IouSimilarity()

        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(scale=float(0.00004)),
            weights_initializer=tf.truncated_normal_initializer(mean=0.03, stddev=0.0),
            activation_fn=tf.nn.relu6,
            normalizer_fn=slim.batch_norm,
            normalizer_params={
                'decay': 0.9997,
                'center': True,
                'scale': True,
                'epsilon': 0.001,
                'is_training': self.is_training}) as sc:
            ssd_box_predictor = ConvolutionalBoxPredictor(
                is_training=self.is_training,
                num_classes=self.num_classes,
                conv_hyperparams=sc,
                min_depth=0,
                max_depth=0,
                num_layers_before_predictor=0,
                use_dropout=False,
                dropout_keep_prob=0.8,
                kernel_size=1,
                box_code_size=4,
                apply_sigmoid_to_scores=False)

        anchor_generator = multiple_grid_anchor_generator.create_ssd_anchors(
            num_layers=6,
            min_scale=0.2,
            max_scale=0.95,
            scales=None,
            aspect_ratios=(1.0, 2.0, 0.5, 3.0, 0.3333))

        image_resizer_fn = functools.partial(
            resize_image,
            new_height=272,
            new_width=480)

        non_max_suppression_fn = functools.partial(
            batch_multiclass_non_max_suppression,
            score_thresh=1e-8,
            iou_thresh=0.6,
            max_size_per_class=100,
            max_total_size=100
        )

        score_conversion_fn = _score_converter_fn_with_logit_scale(tf.sigmoid, logit_scale=1.0)

        classification_loss   = WeightedSigmoidClassificationLoss()
        localization_loss     = WeightedSmoothL1LocalizationLoss()
        classification_weight = self.xent_rate
        localization_weight   = self.iou_rate
        hard_example_miner    = HardExampleMiner(
            num_hard_examples=3000,
            iou_threshold=0.99,
            loss_type='cls',
            cls_loss_weight=classification_weight,
            loc_loss_weight=localization_weight,
            max_negatives_per_positive=3,
            min_negatives_per_image=0)

        normalize_loss_by_num_matches = True

        self.ssd_meta_arch = SSDMetaArch_V2(
            self.is_training,
            anchor_generator,
            ssd_box_predictor,
            box_coder,
            feature_extractor,
            matcher,
            region_similarity_calculator,
            image_resizer_fn,
            non_max_suppression_fn,
            score_conversion_fn,
            classification_loss,
            localization_loss,
            classification_weight,
            localization_weight,
            normalize_loss_by_num_matches,
            hard_example_miner,
            add_summaries=True)

    def _build_loss(self):

        train_batch_size_tensor = tf.constant([self.train_batch_size])
        true_image_shapes_single = tf.constant([272, 480, 3])
        true_image_shapes = tf.reshape(tf.tile(true_image_shapes_single, train_batch_size_tensor),
                                       [train_batch_size_tensor[0], tf.shape(true_image_shapes_single)[0]])

        images      = self.images
        bbox_labels = self.bbox_labels
        cls_labels  = self.cls_labels
        num_boxes   = self.num_boxes

        images.set_shape([self.train_batch_size, 272, 480, 3])
        bbox_labels.set_shape([self.train_batch_size, None, 4])
        cls_labels.set_shape([self.train_batch_size, None])
        num_boxes.set_shape([self.train_batch_size])

        # transforming cls labels to one hot:
        cls_labels_one_hot = tf.one_hot(cls_labels, depth=self.num_classes)

        # slicing tensors
        # groundtruth_boxes_tensor   = tf.map_fn(lambda x: tf.slice(x[0], [0,0], [x[1], 4]),
        #                                        (self.bbox_labels, self.num_boxes), dtype=tf.float32, infer_shape=False)
        # groundtruth_classes_tensor = tf.map_fn(lambda x: tf.slice(x[0], [0,0], [x[1], self.num_classes]),
        #                                        (self.cls_labels, self.num_boxes), dtype=tf.float32, infer_shape=False)
        groundtruth_boxes_list   = []
        groundtruth_classes_list = []
        for i in range(self.train_batch_size):
            groundtruth_boxes_list.append(
                tf.slice(bbox_labels[i], [0,0], [num_boxes[i], 4], name='boxes_tensor_slice_{}'.format(i)))
            groundtruth_classes_list.append(
                tf.slice(cls_labels_one_hot[i], [0,0], [num_boxes[i], self.num_classes], name='cls_tensor_slice_{}'.format(i)))

        groundtruth_masks_list     = None
        groundtruth_keypoints_list = None

        resized_images, true_images_shape = self.ssd_meta_arch.preprocess(tf.to_float(images))
        self.ssd_meta_arch.provide_groundtruth(groundtruth_boxes_list,
                                               groundtruth_classes_list,
                                               groundtruth_masks_list,
                                               groundtruth_keypoints_list)

        prediction_dict = self.ssd_meta_arch.predict(resized_images, true_images_shape)
        losses_dict     = self.ssd_meta_arch.loss(prediction_dict, true_image_shapes)
        for loss_tensor in losses_dict.values():
            tf.losses.add_loss(loss_tensor, loss_collection=LOSSES)

        super(SSDMobileNet, self)._build_loss()

    def _decay(self):
        """L2 weight decay loss."""
        cost = tf.losses.get_regularization_loss()
        return tf.multiply(self.weight_decay_rate, cost)

    def add_fidelity_loss(self):
        pass

    def _set_placeholders(self):
        super(SSDMobileNet, self)._set_placeholders()
        self.images             = tf.placeholder(tf.uint8,   shape=[None, 272, 480, 3], name='images')
        self.bbox_labels        = tf.placeholder(tf.float32, shape=[None, None, 4], name='bbox_labels')
        self.cls_labels         = tf.placeholder(tf.int32,   shape=[None, None], name='cls_labels')
        self.num_boxes          = tf.placeholder(tf.int32,   shape=[None], name='num_boxes')
        self.source_ids         = tf.placeholder(tf.string,  shape=[None], name='source_ids')

        self.eval_images        = tf.placeholder(tf.uint8,   shape=[None, 272, 480, 3], name='eval_images')
        self.eval_bbox_labels   = tf.placeholder(tf.float32, shape=[None, None, 4], name='eval_bbox_labels')
        self.eval_cls_labels    = tf.placeholder(tf.int32,   shape=[None, None], name='eval_cls_labels')
        self.eval_num_boxes     = tf.placeholder(tf.int32,   shape=[None], name='eval_num_boxes')
        self.eval_source_ids    = tf.placeholder(tf.string,  shape=[None], name='eval_source_ids')
