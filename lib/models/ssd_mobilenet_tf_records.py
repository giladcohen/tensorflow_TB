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
from object_detection.meta_architectures.ssd_meta_arch import SSDMetaArch
from lib.base.collections import LOSSES
from object_detection.core import standard_fields as fields
from object_detection.utils import ops as util_ops
from object_detection.core import prefetcher
from object_detection import eval_util

def get_inputs(input_queue, num_classes):
    read_data_list = input_queue.dequeue()
    label_id_offset = 1

    def extract_images_and_targets(read_data):
        """Extract images and targets from the input dict."""
        image = read_data[fields.InputDataFields.image]
        key = ''
        if fields.InputDataFields.source_id in read_data:
            key = read_data[fields.InputDataFields.source_id]
        location_gt = read_data[fields.InputDataFields.groundtruth_boxes]
        classes_gt = tf.cast(read_data[fields.InputDataFields.groundtruth_classes], tf.int32)
        classes_gt -= label_id_offset
        classes_gt = util_ops.padded_one_hot_encoding(indices=classes_gt, depth=num_classes, left_pad=0)
        masks_gt = read_data.get(fields.InputDataFields.groundtruth_instance_masks)
        keypoints_gt = read_data.get(fields.InputDataFields.groundtruth_keypoints)
        weights_gt = read_data.get(fields.InputDataFields.groundtruth_weights)
        return (image, key, location_gt, classes_gt, masks_gt, keypoints_gt, weights_gt)

    return zip(*map(extract_images_and_targets, read_data_list))

def extract_prediction_tensors(model, input_dict):
    """Restores the model in a tensorflow session.

    Args:
    model: model to perform predictions with.
    input_dict: input tensor dictionaries.

    Returns:
    tensor_dict: A tensor dictionary with evaluations.
    """
    # prefetch_queue = prefetcher.prefetch(input_dict, capacity=100)
    # input_dict = prefetch_queue.dequeue()
    original_image = tf.expand_dims(input_dict[fields.InputDataFields.image], 0)
    preprocessed_image, true_image_shapes = model.preprocess(tf.to_float(original_image))
    prediction_dict = model.predict(preprocessed_image, true_image_shapes)
    detections = model.postprocess(prediction_dict, true_image_shapes)

    groundtruth = {
        fields.InputDataFields.groundtruth_boxes:
            input_dict[fields.InputDataFields.groundtruth_boxes],
        fields.InputDataFields.groundtruth_classes:
            input_dict[fields.InputDataFields.groundtruth_classes],
        fields.InputDataFields.groundtruth_area:
            input_dict[fields.InputDataFields.groundtruth_area],
        fields.InputDataFields.groundtruth_is_crowd:
            input_dict[fields.InputDataFields.groundtruth_is_crowd],
        fields.InputDataFields.groundtruth_difficult:
            input_dict[fields.InputDataFields.groundtruth_difficult]
    }
    if fields.InputDataFields.groundtruth_group_of in input_dict:
      groundtruth[fields.InputDataFields.groundtruth_group_of] = (
          input_dict[fields.InputDataFields.groundtruth_group_of])
    if fields.DetectionResultFields.detection_masks in detections:
      groundtruth[fields.InputDataFields.groundtruth_instance_masks] = (
          input_dict[fields.InputDataFields.groundtruth_instance_masks])

    return eval_util.result_dict_for_single_example(
        original_image,
        input_dict[fields.InputDataFields.source_id],
        detections,
        groundtruth,
        class_agnostic=(
            fields.DetectionResultFields.detection_classes not in detections),
        scale_to_absolute=True)

class SSDMobileNet(ModelBase):

    def __init__(self, *args, **kwargs):
        super(SSDMobileNet, self).__init__(*args, **kwargs)
        self.num_classes        = self.prm.network.NUM_CLASSES
        self.images             = None
        self.ssd_meta_arch      = None
        self.train_input_queue  = None
        self.test_input_tensor  = None
        self.tensor_dict        = None

    def print_stats(self):
        super(SSDMobileNet, self).print_stats()
        self.log.info(' NUM_CLASSES: {}'.format(self.num_classes))

    def _build_interpretation(self):
        # input_tensor = self.test_input_tensor
        # input_tensor = {fields.InputDataFields.image:                 input_tensor[0],
        #                 fields.InputDataFields.groundtruth_boxes:     input_tensor[2],
        #                 fields.InputDataFields.groundtruth_classes:   input_tensor[3],
        #                 fields.InputDataFields.groundtruth_is_crowd:  input_tensor[5],
        #                 fields.InputDataFields.groundtruth_area:      input_tensor[6],
        #                 fields.InputDataFields.groundtruth_group_of:  input_tensor[7],
        #                 fields.InputDataFields.groundtruth_difficult: input_tensor[8],
        #                 fields.InputDataFields.source_id:             input_tensor[9]}
        #
        # self.tensor_dict = extract_prediction_tensors(self.ssd_meta_arch, input_tensor)
        pass

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

        self.ssd_meta_arch = SSDMetaArch(
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

        (images, _, groundtruth_boxes_list, groundtruth_classes_list,
         groundtruth_masks_list, groundtruth_keypoints_list, _) = get_inputs(
            self.train_input_queue,
            self.num_classes)

        preprocessed_images = []
        true_image_shapes = []
        for image in images:
            resized_image, true_image_shape = self.ssd_meta_arch.preprocess(image)
            preprocessed_images.append(resized_image)
            true_image_shapes.append(true_image_shape)

        images = tf.concat(preprocessed_images, 0)
        true_image_shapes = tf.concat(true_image_shapes, 0)
        self.images = images

        if any(mask is None for mask in groundtruth_masks_list):
            groundtruth_masks_list = None
        if any(keypoints is None for keypoints in groundtruth_keypoints_list):
            groundtruth_keypoints_list = None

        self.ssd_meta_arch.provide_groundtruth(groundtruth_boxes_list,
                                               groundtruth_classes_list,
                                               groundtruth_masks_list,
                                               groundtruth_keypoints_list)

        prediction_dict = self.ssd_meta_arch.predict(images, true_image_shapes)
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
