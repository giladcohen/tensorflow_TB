import os
import sys

cwd = os.getcwd() # tensorflow-TB
sys.path.insert(0, cwd)

import lib.logger.logger as logger
from lib.logger.logging_config import logging_config
from utils.parameters import Parameters
from utils.factories import Factories
from lib.datasets.dataset_wrapper import DatasetWrapper
import tensorflow as tf

logging = logging_config()

logging.disable(logging.DEBUG)
log = logger.get_logger('main')

prm_file = '/data/gilad/logs/log_2210_220817_wrn-fc2_kmeans_SGD_init_200_clusters_4_cap_204/parameters.ini'

prm = Parameters()
prm.override(prm_file)

dev = prm.network.DEVICE

factories = Factories(prm)

model = factories.get_model()
model.print_stats()  # debug

preprocessor = factories.get_preprocessor()
preprocessor.print_stats()  # debug

train_dataset = factories.get_train_dataset(preprocessor)
validation_dataset = factories.get_validation_dataset(preprocessor)

dataset_wrapper = DatasetWrapper(prm.dataset.DATASET_NAME + '_wrapper', prm, train_dataset, validation_dataset)
dataset_wrapper.print_stats()

# trainer = factories.get_trainer(model, dataset_wrapper)
# trainer.print_stats()  # debug

# start debugging
model.build_graph()
saver = tf.train.Saver(max_to_keep=None, name='debug', filename='model_debug')
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# restore checkpoint
ckpt_state = tf.train.get_checkpoint_state(prm.train.train_control.CHECKPOINT_DIR)





