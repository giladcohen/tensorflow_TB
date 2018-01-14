from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
from lib.base.agent import Agent
import tensorflow as tf
import os
from utils.tensorboard_logging import TBLogger

class TesterBase(Agent):
    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        super(TesterBase, self).__init__(*args, **kwargs)

        self.test_dir              = self.prm.train.train_control.TEST_DIR

        # testing parameters
        self.tester          = self.prm.test.test_control.TESTER         # just used for printing.
        self.checkpoint_file = self.prm.test.test_control.CHECKPOINT_FILE
        self.dump_net        = self.prm.test.test_control.DUMP_NET

    @abstractmethod
    def test(self):
        pass

    def build_test_env(self):
        super(TesterBase, self).build_test_env()
        self.log.info("Starting building the test environment")
        self.summary_writer_test = tf.summary.FileWriter(self.test_dir)
        self.tb_logger_test = TBLogger(self.summary_writer_test)

    def build_session(self):
        super(TesterBase, self).build_session()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.plain_sess = self.sess

    def print_stats(self):
        '''print basic test parameters'''
        self.log.info('Test parameters:')
        super(TesterBase, self).print_stats()
        self.log.info(' TEST_DIR: {}'.format(self.test_dir))
        self.log.info(' TESTER: {}'.format(self.tester))
        self.log.info(' CHECKPOINT_FILE: {}'.format(self.checkpoint_file))
        self.log.info(' DUMP_NET: {}'.format(self.dump_net))

    def finalize_graph(self):
        self.saver.restore(self.plain_sess, os.path.join(self.checkpoint_dir, self.checkpoint_file))
        super(TesterBase, self).finalize_graph()
