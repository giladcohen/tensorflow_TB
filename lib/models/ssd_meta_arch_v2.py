from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from object_detection.meta_architectures.ssd_meta_arch import SSDMetaArch
import tensorflow as tf
from object_detection.utils import shape_utils

class SSDMetaArch_V2(SSDMetaArch):
    def preprocess(self, inputs):
        """Feature-extractor specific preprocessing.

        SSD meta architecture uses a default clip_window of [0, 0, 1, 1] during
        post-processing. On calling `preprocess` method, clip_window gets updated
        based on `true_image_shapes` returned by `image_resizer_fn`.

        Args:
          inputs: a [batch, height_in, width_in, channels] float tensor representing
            a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: a [batch, height_out, width_out, channels] float
            tensor representing a batch of images.
          true_image_shapes: int32 tensor of shape [batch, 3] where each row is
            of the form [height, width, channels] indicating the shapes
            of true images in the resized images, as resized images can be padded
            with zeros.

        Raises:
          ValueError: if inputs tensor does not have type tf.float32
        """
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')
        with tf.name_scope('Preprocessor'):
            # TODO: revisit whether to always use batch size as
            # the number of parallel iterations vs allow for dynamic batching.
            outputs = shape_utils.static_or_dynamic_map_fn(
                self._image_resizer_fn,
                elems=inputs,
                dtype=[tf.float32, tf.int32])
            resized_inputs = outputs[0]
            true_image_shapes = outputs[1]

            resized_inputs = tf.map_fn(tf.image.per_image_standardization, resized_inputs, dtype=tf.float32, parallel_iterations=10)

            return (self._feature_extractor.preprocess(resized_inputs),
                    true_image_shapes)

