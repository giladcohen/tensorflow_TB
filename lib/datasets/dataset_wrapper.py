import lib.logger.logger as logger
from lib.datasets.dataset import DataSet

class DatasetWrapper(object):
    """Wrapper which hold both the trainset and the validation set"""

    def __init__(self, name, prm, preprocessor):
        self.name = name
        self.prm = prm
        self.preprocessor = preprocessor
        self.log = logger.get_logger(name)

        self.dataset_name           = self.prm.dataset.DATASET_NAME
        self.train_set_size         = self.prm.dataset.TRAIN_SET_SIZE
        self.validation_set_size    = self.prm.dataset.VALIDATION_SET_SIZE
        self.train_images_dir       = self.prm.dataset.TRAIN_IMAGES_DIR
        self.train_labels_file      = self.prm.dataset.TRAIN_LABELS_FILE
        self.validation_images_dir  = self.prm.dataset.VALIDATION_IMAGES_DIR
        self.validation_labels_file = self.prm.dataset.VALIDATION_LABELS_FILE

        self.train_dataset      = DataSet(self.name + '_train'     , self.prm, self.preprocessor)
        self.validation_dataset = DataSet(self.name + '_validation', self.prm, self.preprocessor)

    def __str__(self):
        return self.name

    def print_stats(self):
        """print dataset parameters"""
        self.log.info('Dataset parameters:')
        self.log.info(' DATASET_NAME: {}'.format(self.dataset_name))
        self.log.info(' TRAIN_SET_SIZE: {}'.format(self.train_set_size))
        self.log.info(' VALIDATION_SET_SIZE: {}'.format(self.validation_set_size))
        self.log.info(' TRAIN_IMAGES_DIR: {}'.format(self.train_images_dir))
        self.log.info(' TRAIN_LABELS_FILE: {}'.format(self.train_labels_file))
        self.log.info(' VALIDATION_IMAGES_DIR: {}'.format(self.validation_images_dir))
        self.log.info(' VALIDATION_LABELS_FILE: {}'.format(self.validation_labels_file))

    def get_mini_batch_train(self, *args, **kwargs):
        return self.train_dataset.get_mini_batch(*args, **kwargs)

    def get_mini_batch_validate(self, *args, **kwargs):
        return self.validation_dataset.get_mini_batch(*args, **kwargs)
