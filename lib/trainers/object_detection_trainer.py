from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from lib.trainers.trainer_base import TrainerBase
from lib.base.collections import TRAIN_SUMMARIES
import numpy as np
import os
from utils.tensorboard_logging import TBLogger
from lib.trainers.hooks.global_step_checkpoint_saver_hook import GlobalStepCheckpointSaverHook
from object_detection import eval_util
from object_detection.utils import label_map_util
from object_detection.utils.object_detection_evaluation import PascalDetectionEvaluator
from tensorflow.python import debug as tf_debug
from object_detection.core import standard_fields as fields

class ObjectDetectionTrainer(TrainerBase):
    """Implementing object detection trainer
    Using the entire labeled trainset for training"""

    def __init__(self, *args, **kwargs):
        super(ObjectDetectionTrainer, self).__init__(*args, **kwargs)
        # self.categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes)
        self.categories = [{'id': 1, 'name': 'car'},
                           {'id': 2, 'name': 'pickup_truck'},
                           {'id': 3, 'name': 'truck'},
                           {'id': 4, 'name': 'bus'},
                           {'id': 5, 'name': 'van'},
                           {'id': 6, 'name': 'motorcycle'},
                           {'id': 7, 'name': 'bicycle'}]

        self.evaluator = PascalDetectionEvaluator(categories=self.categories)
        self.num_batches = 10

    def train_step(self):
        '''Implementing one training step'''
        print('DEBUG: just about to train with global step {}'.format(self.global_step))
        image_fnames, images, bbox_labels, cls_labels, num_boxes = self.dataset.get_mini_batch('train', self.plain_sess)
        _ , self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
                                             feed_dict={self.model.source_ids : image_fnames,
                                                        self.model.images     : images,
                                                        self.model.bbox_labels: bbox_labels,
                                                        self.model.cls_labels : cls_labels,
                                                        self.model.num_boxes  : num_boxes,
                                                        self.model.eval_source_ids: np.expand_dims(image_fnames[0], axis=0),
                                                        self.model.eval_images: np.expand_dims(images[0], axis=0),
                                                        self.model.eval_bbox_labels: np.expand_dims(bbox_labels[0], axis=0),
                                                        self.model.eval_cls_labels: np.expand_dims(cls_labels[0], axis=0),
                                                        self.model.eval_num_boxes: np.expand_dims(num_boxes[0], axis=0),
                                                        self.model.is_training: True})

    def test_step(self):
        '''Implementing one test step.'''
        self.log.info('start running test within training. global_step={}'.format(self.global_step))
        self.plain_sess.run(self.dataset.test_iterator.initializer)

        counters = {'skipped': 0, 'success': 0}
        try:
            for batch in range(int(self.num_batches)):
                image_fnames, images, bbox_labels, cls_labels, num_boxes = self.dataset.get_mini_batch('test', self.plain_sess)
                feed_dict = {self.model.eval_source_ids: image_fnames,
                             self.model.eval_images: images,
                             self.model.eval_bbox_labels: bbox_labels,
                             self.model.eval_cls_labels: cls_labels,
                             self.model.eval_num_boxes: num_boxes,
                             self.model.is_training: False}
                result_dict = self.process_batch(self.model.tensor_dict, self.plain_sess, batch, counters, feed_dict)
                if not result_dict:
                    continue
                self.evaluator.add_single_ground_truth_image_info(image_id=batch, groundtruth_dict=result_dict)
                self.evaluator.add_single_detected_image_info(image_id=batch, detections_dict=result_dict)
        except tf.errors.OutOfRangeError:
            self.log.info('Done evaluating -- epoch limit reached')
        finally:
            self.log.info('# success: %d', counters['success'])
            self.log.info('# skipped: %d', counters['skipped'])
            all_evaluator_metrics = {}
            metrics = self.evaluator.evaluate()
            self.evaluator.clear()
            all_evaluator_metrics.update(metrics)

        eval_util.write_metrics(all_evaluator_metrics, self.global_step, self.test_dir)

    def eval_step(self):
        pass

    def get_train_summaries(self):
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('weight_decay_rate', self.model.weight_decay_rate))
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.image('input_images'      , self.model.images))

    def set_params(self):
        super(ObjectDetectionTrainer, self).set_params()
        assign_ops = []
        iou_rate = self.plain_sess.run(self.model.iou_rate)

        if not np.isclose(iou_rate, self.prm.network.optimization.IOU_RATE):
            assign_ops.append(self.model.assign_ops['iou_rate'])
            self.log.warning('changing model.iou_rate from {} to {}'.format(iou_rate, self.prm.network.optimization.IOU_RATE))

        self.plain_sess.run(assign_ops)

    def build_train_env(self):
        self.log.info("Starting building the train environment")
        self.summary_writer_train = tf.summary.FileWriter(self.train_dir)
        self.tb_logger_train = TBLogger(self.summary_writer_train)
        self.get_train_summaries()  #FIXME(gilad): change to "set_train_summaries"

        self.learning_rate_hook = self.Factories.get_learning_rate_setter(self.model, self.test_retention)

        summary_hook = tf.train.SummarySaverHook(
            save_steps=self.summary_steps,
            summary_writer=self.summary_writer_train,
            summary_op=tf.summary.merge([self.model.summaries] + tf.get_collection(TRAIN_SUMMARIES))
        )

        logging_hook = tf.train.LoggingTensorHook(
            tensors={'step': self.model.global_step,
                     'loss-loss_wd': self.model.cost - self.model.wd_cost,
                     'loss_wd': self.model.wd_cost,
                     'loss': self.model.cost},
            every_n_iter=self.logger_steps)

        checkpoint_hook = GlobalStepCheckpointSaverHook(name='global_step_checkpoint_saver_hook',
                                                        prm=self.prm,
                                                        model=self.model,
                                                        steps_to_save=self.checkpoint_steps,
                                                        checkpoint_dir=self.checkpoint_dir,
                                                        saver=self.saver,
                                                        checkpoint_basename='model_schedule.ckpt')

        stop_at_step_hook = tf.train.StopAtStepHook(last_step=self.last_step)

        # tfdbg_hook = tf_debug.LocalCLIDebugHook()

        # self.train_session_hooks = [summary_hook, logging_hook, self.learning_rate_hook, checkpoint_hook, stop_at_step_hook, tfdbg_hook]
        # FIXME(gilad): switch to dictionary instead of list
        self.train_session_hooks = [summary_hook, logging_hook, self.learning_rate_hook, checkpoint_hook, stop_at_step_hook]

    def finalize_graph(self):
        print('DEBUG: start finalizing {}'.format(self.global_step))
        # self.plain_sess.run(self.dataset.test_iterator.initializer)
        # print('DEBUG: initialized test_iterator')
        # self.plain_sess.run(self.dataset.train_iterator.initializer)
        # print('DEBUG: initialized train_iterator')
        self.dataset.set_handles(self.plain_sess)
        self.global_step = self.plain_sess.run(self.model.global_step)
        print('DEBUG: global step init to {}'.format(self.global_step))

    def process_batch(self, tensor_dict, sess, batch_index, counters, feed_dict):
        """Evaluates tensors in tensor_dict, visualizing the first K examples.

        This function calls sess.run on tensor_dict, evaluating the original_image
        tensor only on the first K examples and visualizing detections overlaid
        on this original_image.

        Args:
          tensor_dict: a dictionary of tensors
          sess: tensorflow session
          batch_index: the index of the batch amongst all batches in the run.
          counters: a dictionary holding 'success' and 'skipped' fields which can
            be updated to keep track of number of successful and failed runs,
            respectively.  If these fields are not updated, then the success/skipped
            counter values shown at the end of evaluation will be incorrect.
          feed_dict: the feed_dict which corresponds to the session.run call

        Returns:
          result_dict: a dictionary of numpy arrays
        """
        try:
            result_dict = sess.run(tensor_dict, feed_dict=feed_dict)
            counters['success'] += 1
        except tf.errors.InvalidArgumentError:
            self.log.info('Skipping image')
            counters['skipped'] += 1
            return {}
        result_dict[fields.InputDataFields.key] = os.path.basename(result_dict[fields.InputDataFields.key])
        global_step = tf.train.global_step(sess, tf.train.get_global_step())
        if batch_index < 10:
            tag = 'image-{}'.format(batch_index)
            eval_util.visualize_detection_results(
                result_dict,
                tag,
                global_step,
                categories=self.categories,
                summary_dir=self.test_dir,
                export_dir=self.pred_dir,
                show_groundtruth=True,
                groundtruth_box_visualization_color='black',
                min_score_thresh=0.5,
                max_num_predictions=20,
                skip_scores=False,
                skip_labels=False,
                keep_image_id_for_visualization_export=True)
        return result_dict

