import numpy as np
from lib.preprocessors.preprocessor_base import PreProcessorBase
from utils.image_utils import scale_img

expected_scale = ()

class ParkProcessor(PreProcessorBase):

    def process(self, images, labels):
        """processing batch of images and labels"""
        super(ParkProcessor, self).process(images, labels)
        # There is no need for label augmentation for classification task
        # Therefore self.label_augmentation is not used here. It will be used for preprocessing
        # other tasks such as detection/segmentation
        images_aug = np.empty(shape=[images.shape[0],
                                     self.prm.network.IMAGE_HEIGHT,
                                     self.prm.network.IMAGE_WIDTH,
                                     3],
                              dtype=np.uint8)
        labels_aug = np.empty(shape=[len(labels)], dtype=np.int)

        for i in xrange(images.shape[0]):
            image_orig = images[i]
            label      = labels[i]

            # scaling
            image, scale, orig_dims = scale_img(image_orig, self.prm.network.IMAGE_WIDTH, self.prm.network.IMAGE_HEIGHT)
            assert orig_dims == (1280, 720), 'The original dims are ({},{})'.format(orig_dims[0], orig_dims[1])
            H, W, D = image.shape

            # augmentation
            min_drift_x = -self.drift_x
            max_drift_x =  self.drift_x
            min_drift_y = -self.drift_y
            max_drift_y =  self.drift_y
            dy = self.rand_gen.randint(min_drift_y, max_drift_y + 1)
            dx = self.rand_gen.randint(min_drift_x, max_drift_x + 1)

            orig_x, dist_x = max(dx, 0), max(-dx, 0)
            orig_y, dist_y = max(dy, 0), max(-dy, 0)
            distorted_im = np.zeros([H, W, D])
            distorted_im[dist_y:H - orig_y, dist_x:W - orig_x, :] = image[orig_y:H - dist_y, orig_x:W - dist_x, :]
            image = distorted_im

            if self.flip_image and self.rand_gen.randint(2) > 0.5:
                image = image[:, ::-1, :]

            images_aug[i] = image
            labels_aug[i] = label  # for classification task

        return images_aug, labels_aug
