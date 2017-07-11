import numpy as np
import random
import os
import re
from PIL import Image
from lib.preprocessor import PreProcessor

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class DataTank(object):

    def __init__(self, data_path, label_file, batch_size, N, to_preprocess=True):
        self.data_path  = data_path
        self.label_file = label_file
        self.batch_size = batch_size
        self.H          = 32
        self.W          = 32
        self.N          = N
        self.pool       = range(N)
        self.data_list  = self.create_data_list()
        self.labels     = self.create_labels()
        self.preprocessor = PreProcessor(to_preprocess)
        assert len(self.data_list) == N
        assert len(self.data_list) == len(self.labels)

    def create_data_list(self):
        data_list = []
        local_list = sorted(os.listdir(self.data_path), key=numericalSort)
        for file in local_list:
            data_list.append(os.path.join(self.data_path, file))
        assert len(data_list) == len(local_list)
        return data_list

    def create_labels(self):
        labels = -1 * np.ones([self.N], dtype=np.int)
        tmp_list = open(self.label_file).read().splitlines()
        for i, val in enumerate(tmp_list):
            labels[i] = int(val)
        assert np.sum(labels == -1) == 0
        return labels

    def fetch_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        indices = np.random.choice(self.pool, batch_size)
        return self.fetch_batch_common(indices)

    def fetch_batch_common(self, indices):
        batch_size = len(indices)
        images = np.empty([batch_size, self.H, self.W, 3], np.uint8)
        labels = self.labels[indices]
        for i in xrange(batch_size):
            image_file = self.data_list[indices[i]]
            images[i] = np.asarray(Image.open(image_file), dtype=np.uint8)
        images_aug, labels_aug = self.preprocessor.process(images, labels)
        return images, labels, images_aug, labels_aug
