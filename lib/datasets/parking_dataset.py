import os
import numpy as np
import json
from lib.datasets.dataset_base import DataSetBase
from utils.parsing import parse_images_params_str, get_full_names_of_image_files
from utils.image_utils import img_to_mat
import utils.fonts

class ParkingDataSet(DataSetBase):

    def set_additional_config(self):
        self.H = self.prm.network.pre_processing.IMAGE_HEIGHT
        self.W = self.prm.network.pre_processing.IMAGE_WIDTH
        self.clusters  = self.prm.dataset.CLUSTERS
        self.cap       = self.prm.dataset.CAP  # must not be None

        self.num_classes = self.prm.network.NUM_CLASSES
        self.images_dir_file = self.images_dir  # for this dataset we use the notation <dir>:<images.txt>
        self.tagged_images_dir_file = self.prm.dataset.TAGGED_IMAGES_DIR

        if 'train' in self.name:
            self.labels_dir = self.prm.dataset.TRAIN_LABELS_DIR
        elif 'validation' in self.name:
            self.labels_dir = self.prm.dataset.VALIDATION_LABELS_DIR

        self.create_images_list()
        self.create_labels_dict()

    def assert_config(self):
        if self.cap is None:
            err_str = self.__str__() + ': CAP cannot be None'
            self.log.error(err_str)
            raise AssertionError(err_str)
        if self.init_size is None:
            self.log.warning(self.__str__() + 'Initialized with INIT_SIZE=None. Setting INIT_SIZE=CAP ({})'.format(self.cap))
            self.init_size = self.cap

    def create_images_list(self):
        """creates list of images in the dataset:
        self.image_fnames
        self.base_image_fnames
        """
        try:
            self.images_dir, self.images_file        = parse_images_params_str(self.images_dir_file)
            _              , self.tagged_images_file = parse_images_params_str(self.tagged_images_dir_file)
        except:
            err_str = 'Failed to parse image directory from parameter: {}'.format(self.images_dir_file)
            self.log.exception(err_str)
            raise AssertionError(err_str)

        try:
            self.image_fnames = get_full_names_of_image_files(self.images_dir, self.images_file)
            self.image_fnames.sort()
            self.base_image_fnames = [os.path.basename(im) for im in self.image_fnames]
            self.tagged_image_fnames = get_full_names_of_image_files(self.images_dir, self.tagged_images_file)
            self.tagged_image_fnames.sort()
            self.base_tagged_image_fnames = [os.path.basename(im) for im in self.tagged_image_fnames]
        except:
            err_str = 'Failed to load image list of image files'
            self.log.exception(err_str)
            raise Exception(err_str)

    def create_labels_dict(self):
        """creates dictionary of images in the dataset with key of base file name
        self.labels_dict
        """
        if not os.path.isdir(self.labels_dir):
            err_str = 'labels directory not found: {}'.format(self.labels_dir)
            self.log.error(err_str)
            raise AssertionError(err_str)
        self.labels_dict = self.read_labels(self.base_image_fnames, self.labels_dir)

    def read_labels(self, image_base_filenames, labels_dir):
        idx2annotation = {}
        for img_fname in image_base_filenames:
            labels_fname = os.path.join(labels_dir, img_fname + '.json')
            if not os.path.isfile(labels_fname):
                err_str = 'label filename {} does not exist'.format(labels_fname)
                self.log.error(err_str)
                raise AssertionError(err_str)
            else:
                with open(labels_fname) as infile:
                    annotations = json.load(infile)
            labels = {}
            labels['gps_coord']      = (annotations['gps_coord_x'], annotations['gps_coord_y'])
            labels['city']           =  annotations['city']
            labels['park_available'] =  annotations['park_available']
            idx2annotation[img_fname] = labels
        return idx2annotation

    def get_mini_batch(self, batch_size=None, indices=None):
        if indices is not None:
            return self.indices_to_data(indices)
        if batch_size is None:
            batch_size = self.batch_size
        indices = self.minibatch_server.get_mini_batch(batch_size)
        return self.indices_to_data(indices)

    def indices_to_data(self, indices):
        batch_size = len(indices)
        images_out = np.empty([batch_size, self.H, self.W, 3], np.uint8)
        labels_out = -1 * np.ones([batch_size], np.int32)

        for i in range(batch_size):
            index = indices[i]
            image_file      = self.image_fnames[index]
            base_image_file = self.base_image_fnames[index]
            label           = self.labels_dict[base_image_file]['parking_available']
            image  = img_to_mat(image_file, as_rgb=True)
            image, scale, labels = self.preprocessor.process(image, label, with_augmentation=self.to_preprocess)
            images_out[i] = image
            labels_out[i] = labels
        return images_out, labels_out

    def initialize_pool(self):
        self.log.info('Initializing pool with {} initial values'.format(len(self.base_tagged_image_fnames)))
        self.pool = []
        self.available_samples = range(self.size)
        indices = []
        for ind, val in enumerate(self.base_image_fnames):
            if val in self.base_tagged_image_fnames:
                indices.append(ind)
        self.update_pool(indices=indices)

    def update_pool(self, indices):
        """Indices must or list"""
        self.update_pool_with_indices(indices)

    def update_pool_with_indices(self, indices):
        """indices must be of type list"""
        self.assert_unique_indices(indices)  # time consuming.
        self.pool += indices
        self.pool = sorted(self.pool)
        self.available_samples = [i for j, i in enumerate(self.available_samples) if i not in self.pool]
        self.minibatch_server.set_pool(self.pool)
        self.log.info('updated pool length to {}'.format(self.pool_size()))
        if self.pool_size() > self.cap:
            err_str = 'update_pool_with_indices: pool size ({}) surpassed cap ({})'.format(self.pool_size(), self.cap)
            self.log.error(err_str)
            raise AssertionError(err_str)

    def assert_unique_indices(self, indices):
        for index in indices:
            if index in self.pool:
                err_str = 'update_pool_with_indices: index {} is already in pool.'.format(index)
            if index not in self.available_samples:
                err_str = 'update_pool_with_indices: index {} is not in available_samples.'.format(index)
            if 'err_str' in locals():
                self.log.error(err_str)
                raise AssertionError(err_str)

    # def get_park_list(self, labels_dict):
    #    labels_list = []
    #    for element in labels_dict:
    #        labels_list.append(element['park_available'])
    #     return labels_list
