from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.most_uncertained_trainer import MostUncertainedTrainer
import numpy as np
import tensorflow as tf
from lib.base.collections import TRAIN_SUMMARIES

import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
font = ImageFont.truetype("utils/fonts/OpenSans-Bold.ttf", 16)


class ParkingTrainer(MostUncertainedTrainer):

    def convert_label_to_string(self, label):
        """
        :param label: 0/1
        :return: returns "Parking"
        """
        ret = None
        if label == 0:
            ret = "No Parking"
        elif label == 1:
            ret = "Parking"
        elif label == -1:
            raise AssertionError('convert_label_to_string got unexpected value of label=-1')
        if ret is None:
            raise AssertionError('convert_label_to_string got illegal value of label={}'.format(label))
        return ret

    def overlay_parking_labels(self, images, labels):
        """
        :param images: numpy array of batch of images
        :return: numpy array of batch of images, with overlay labels
        """
        # img_file = '/test_dl/data/NEXET_2017/raw/train/images/incident-0-fd89424b-3b4b-4e27-9a3d-b0e0c77f2140.mp4-0001.jpg'
        # img = cv2.imread(img_file)
        # img = img[:, :, ::-1]

        images_overlayed = np.empty(shape=images.shape, dtype=images.dtype)
        for i in xrange(images.shape[0]):
            img = images[i]
            img_pil = Image.fromarray(img, 'RGB')
            draw = ImageDraw.Draw(img_pil)
            label = labels[i]
            strr = self.convert_label_to_string(label)
            draw.text((0, 0), strr, (255, 0, 0), font=font)
            # img_pil.save('sample-out.jpg')
            images_overlayed[i] = np.asarray(img_pil, dtype=np.uint8)
        return images_overlayed

    def get_train_summaries(self):
        super(ParkingTrainer, self).get_train_summaries()

    def train_step(self):
        '''Implementing one training step'''
        images, labels = self.dataset.get_mini_batch_train()
        _ , self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
                                             feed_dict={self.model.images: images,
                                                        self.model.labels: labels,
                                                        self.model.is_training: True})
        images_overlayed = self.overlay_parking_labels(images, labels)
        max_overlayed_images = max(10, images_overlayed.shape[0])
        self.tb_logger_train.log_images('overlays', images_overlayed[max_overlayed_images], self.global_step)

    def eval_step(self):
        '''Implementing one evaluation step.'''
        self.log.info('start running eval within training. global_step={}'.format(self.global_step))
        total_samples, total_score = 0, 0
        for i in range(self.eval_batch_count):
            b = i * self.eval_batch_size
            if i < (self.eval_batch_count - 1) or (self.last_eval_batch_size == 0):
                e = (i + 1) * self.eval_batch_size
            else:
                e = i * self.eval_batch_size + self.last_eval_batch_size
            images, labels = self.dataset.get_mini_batch_validate(indices=range(b, e))
            (summaries, loss, train_step, predictions) = self.sess.run(
                [self.model.summaries, self.model.cost,
                 self.model.global_step, self.model.predictions],
                feed_dict={self.model.images     : images,
                           self.model.labels     : labels,
                           self.model.is_training: False})

            total_score   += np.sum(labels == predictions)
            total_samples += images.shape[0]
        if total_samples != self.dataset.validation_dataset.size:
            self.log.error('total_samples equals {} instead of {}'.format(total_samples, self.dataset.validation_set.size))
        score = total_score / total_samples
        self.retention.add_score(score, train_step)
        overlays = self.overlay_parking_labels(images, predictions)

        self.tb_logger_eval.log_scalar('score', score, train_step)
        self.tb_logger_eval.log_scalar('best score', self.retention.get_best_score(), train_step)
        self.tb_logger_eval.log_images('overlays', overlays, train_step)
        self.summary_writer_eval.add_summary(summaries, train_step)
        self.summary_writer_eval.flush()
        self.log.info('EVALUATION (step={}): loss: {}, score: {}, best score: {}' \
                      .format(train_step, loss, score, self.retention.get_best_score()))

