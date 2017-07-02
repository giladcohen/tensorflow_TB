import numpy as np


class PreProcessor(object):

    def __init__(self, to_preprocess):
        self.to_preprocess = to_preprocess
        self._drift_x = 4
        self._drift_y = 4

    def process(self, images, labels):
        assert (images.shape[0] == labels.shape[0])
        H, W , D = images.shape[1:4]
        if self.to_preprocess:
            images_aug = np.empty(images.shape, dtype=np.uint8)
            labels_aug = np.empty(labels.shape, dtype=np.int)
            for i in xrange(images.shape[0]):
                image = images[i]
                label = labels[i]
                min_drift_x = -self._drift_x
                max_drift_x =  self._drift_x
                min_drift_y = -self._drift_y
                max_drift_y =  self._drift_y
                dy = np.random.randint(min_drift_y, max_drift_y + 1)
                dx = np.random.randint(min_drift_x, max_drift_x + 1)

                orig_x, dist_x = max(dx, 0), max(-dx, 0)
                orig_y, dist_y = max(dy, 0), max(-dy, 0)
                distorted_im = np.zeros([H, W, D])
                distorted_im[dist_y:H - orig_y, dist_x:W - orig_x, :] = \
                       image[orig_y:H - dist_y, orig_x:W - dist_x, :]

                image = distorted_im

                if np.random.randint(2) > 0.5:
                    # print('flipping for index i=%0d' %i)
                    image = image[:, ::-1, :]

                images_aug[i] = image
                labels_aug[i] = label #for classification task

        else:
            images_aug = images
            labels_aug = labels

        return images_aug, labels_aug
