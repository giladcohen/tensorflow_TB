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
        self.images_list = None
        self.labels = None

    def get_mini_batch(self, batch_size=None, indices=None):
        if indices is not None:
            return self.indices_to_data(indices)
        if batch_size is None:
            batch_size = self.batch_size

        indices = self.minibatch_server.get_mini_batch(batch_size)
        return self.indices_to_data(indices)

    def indices_to_data(self, indices):
        batch_size = len(indices)
        images = np.empty([batch_size, self.H, self.W, 3], np.uint8)
        labels = self.labels[indices]
        for i in xrange(batch_size):
            image_file = self.images_list[indices[i]]
            images[i] = np.asarray(Image.open(image_file), dtype=np.uint8)
        if self.to_preprocess:
            images, labels = self.preprocessor.process(images, labels)
        return images, labels

    def save_pool_data(self, file_name):
        """
        :param file_name: File name prefix to save the pool info into it
        :return: no return.
        """
        images, labels = self.indices_to_data(self.pool)
        np.savetxt(file_name + '_pool.txt', self.pool, fmt='%0d')
        np.save(file_name + '_images.npy', images)
        np.savetxt(file_name + '_labels.txt', labels, fmt='%0d')

