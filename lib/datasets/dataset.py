import numpy as np
import os
import re
from PIL import Image
from lib.datasets.dataset_base import DataSetBase

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class DataSet(DataSetBase):
    def __init__(self, *args, **kwargs):
        super(DataSet, self).__init__(*args, **kwargs)
        self.H = self.prm.network.IMAGE_HEIGHT
        self.W = self.prm.network.IMAGE_WIDTH
        self.images_list = self.create_images_list()
        self.labels = self.create_labels()

    def create_images_list(self):
        images_list = []
        local_list = sorted(os.listdir(self.images_dir), key=numericalSort)
        for file in local_list:
            images_list.append(os.path.join(self.images_dir, file))
        if len(images_list) != self.size:
            err_str = self.__str__() + ': create_data_list: ' + \
                      'number of images ({}) does not match self.size ({})'.format(len(images_list), self.size)
            self.log.error(err_str)
            raise AssertionError(err_str)
        return images_list

    def create_labels(self):
        labels = -1 * np.ones([self.size], dtype=np.int)
        tmp_list = open(self.labels_file).read().splitlines()
        if len(tmp_list) != self.size:
            err_str = self.__str__() + ': create_labels: ' + \
                      'number of labels ({}) does not match self.size ({})'.format(len(tmp_list), self.size)
            self.log.error(err_str)
            raise AssertionError(err_str)
        for i, val in enumerate(tmp_list):
            labels[i] = int(val)
        if np.sum(labels == -1) > 0:
            err_str = self.__str__() + ': create_labels: Some labels contain -1 value. This should not happen'
            self.log.error(err_str)
            raise AssertionError(err_str)
        return labels

    def get_mini_batch(self, batch_size=None, indices=None):
        if indices is not None:
            return self.get_indices(indices)
        if batch_size is None:
            batch_size = self.batch_size
        if self.stochastic:
            indices = self.rand_gen.choice(self.pool, batch_size, replace=False)
        else:
            indices = self.minibatch_server.get_mini_batch(batch_size)
        return self.get_indices(indices)

    def get_indices(self, indices):
        batch_size = len(indices)
        images = np.empty([batch_size, self.H, self.W, 3], np.uint8)
        labels = self.labels[indices]
        for i in xrange(batch_size):
            image_file = self.images_list[indices[i]]
            images[i] = np.asarray(Image.open(image_file), dtype=np.uint8)
        if self.to_preprocess:
            images, labels = self.preprocessor.process(images, labels)
        return images, labels
