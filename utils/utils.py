'''This code converts a numpy image to .bin in the same format of cifar10'''
from __future__ import division

import numpy as np
from keras.datasets import cifar10
import cv2
import os
import lib.logger.logger as logger
from lib.preprocessors.preprocessor import PreProcessor
from lib.trainers.passive_trainer import PassiveTrainer
from lib.models.resnet_model import ResNet
from lib.datasets.dataset_wrapper import DatasetWrapper

def convert_numpy_to_bin(images, labels, save_file, h=32, w=32):
    """Converts numpy data in the form:
    images: [N, H, W, D]
    labels: [N]
    to a .bin file in a CIFAR10 protocol
    """
    images = (np.array(images))
    N = images.shape[0]
    record_bytes = 3 * h * w + 1 #includes also the label
    out = np.zeros([record_bytes * N], np.uint8)
    for i in range(N):
        im = images[i]
        r = im[:,:,0].flatten()
        g = im[:,:,1].flatten()
        b = im[:,:,2].flatten()
        label = labels[i]
        out[i*record_bytes:(i+1)*record_bytes] = np.array(list(label) + list(r) + list(g) + list(b), np.uint8)
    out.tofile(save_file)

def save_cifar10_to_disk(train_data_dir, train_labels_file, test_data_dir, test_labels_file):
    """Saving CIFAR10 train/test data to specified dirs
       Saving CIFAR10 train/test labels to specified files"""
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    np.savetxt(train_labels_file, Y_train, fmt='%0d')
    np.savetxt(test_labels_file,  Y_test,  fmt='%0d')
    for i in range(X_train.shape[0]):
        img = X_train[i]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(train_data_dir, 'train_image_%0d.png' % i), img_bgr)
    for i in range(X_test.shape[0]):
        img = X_test[i]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(test_data_dir,  'test_image_%0d.png'  % i), img_bgr)

class Factories(object):
    """Class which encapsulate all factories in the TB"""

    def __init__(self, prm):
        self.log = logger.get_logger('factories')
        self.prm = prm

        self.dataset = self.prm.dataset.DATASET_NAME
        self.preprocessor = self.prm.network.pre_processing.PREPROCESSOR
        self.trainer = prm.train.train_control.TRAINER
        self.architecture = prm.network.ARCHITECTURE

    def get_dataset(self, preprocessor):
        available_datasets = {'cifar10': DatasetWrapper, 'cifar100': DatasetWrapper}
        if self.dataset in available_datasets:
            dataset = available_datasets[self.dataset](self.dataset, self.prm, preprocessor)
            self.log.info('get_dataset: returning ' + str(dataset))
            return dataset
        else:
            err_str = 'get_dataset: dataset {} was not found. Available datasets are: {}'.format(self.dataset, available_datasets.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_model(self):
        available_networks = {'Wide-Resnet-28-10': ResNet}
        if self.architecture in available_networks:
            model = available_networks[self.architecture](self.architecture, self.prm)
            self.log.info('get_model: returning ' + str(model))
            return model
        else:
            err_str = 'get_model: model {} was not found. Available models are: {}'.format(self.architecture, available_networks.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_preprocessor(self):
        available_preprocessors = {'preprocessor_drift_flip': PreProcessor}
        if self.preprocessor in available_preprocessors:
            preprocessor = available_preprocessors[self.preprocessor](self.preprocessor, self.prm)
            self.log.info('get_preprocessor: returning ' + str(preprocessor))
            return preprocessor
        else:
            err_str = 'get_preprocessor: preprocessor {} was not found. Available preprocessors are: {}'.format(self.preprocessor, available_preprocessors.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)

    def get_trainer(self, model, dataset):
        available_trainers = {'passive': PassiveTrainer}
        if self.trainer in available_trainers:
            trainer = available_trainers[self.trainer](self.trainer, self.prm, model, dataset)
            self.log.info('get_trainer: returning ' + str(trainer))
            return trainer
        else:
            err_str = 'get_trainer: trainer {} was not found. Available trainers are: {}'.format(self.trainer, available_trainers.keys())
            self.log.error(err_str)
            raise AssertionError(err_str)
