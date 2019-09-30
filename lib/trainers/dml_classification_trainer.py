from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_TB.lib.trainers.classification_trainer import ClassificationTrainer
from tensorflow_TB.lib.trainers.hooks.margin_multiplier_setter import MarginMultiplierSetter
from tensorflow_TB.utils.misc import get_vars, collect_features_1d
from tensorflow_TB.lib.base.collections import TRAIN_SUMMARIES
import tensorflow as tf
import os
import numpy as np

class DMLClassificationTrainer(ClassificationTrainer):
    """Implementing classification trainer for DML"""

    def __init__(self, *args, **kwargs):
        super(DMLClassificationTrainer, self).__init__(*args, **kwargs)
        self.checkpoint_ref = self.prm.train.train_control.CHECKPOINT_REF

    def print_stats(self):
        super(DMLClassificationTrainer, self).print_stats()
        self.log.info(' CHECKPOINT_REF: {}'.format(self.checkpoint_ref))

    def build_train_env(self):
        super(DMLClassificationTrainer, self).build_train_env()

        # optionally load checkpoint reference
        if self.checkpoint_ref is not None:
            # first, initialize all variables in the new graph
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))  # use DML graph
            self.log.info('checkpoint_ref was given. Initializing all weight in graph')
            sess.run(self.model.init_op)
            self.log.info('loading to graph pretrained checkpoint file from ref: {}'.format(self.checkpoint_ref))
            vars_to_ignore, vars_to_load = get_vars(tf.global_variables(),
                                                    'RMSProp',
                                                    'init_set_params',
                                                    'fully_connected')
            init_saver = tf.train.Saver(var_list=vars_to_load, name='init_saver', filename='model_ref')
            init_saver.restore(sess, self.checkpoint_ref)
            self.log.info('writing graph with all variables to current checkpoint dir {}'.format(self.checkpoint_dir))
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            self.saver.save(sess, os.path.join(self.checkpoint_dir, 'model.ckpt'))

        self.margin_multiplier_hook = MarginMultiplierSetter(
            name='margin_multiplier_setter',
            prm=self.prm,
            model=self.model,
            retention=self.validation_retention)

        self.train_session_hooks.append(self.margin_multiplier_hook)

    def eval_step(self):
        '''Implementing one evaluation step.'''
        self.log.info('start running eval within training. global_step={}'.format(self.global_step))
        (score,) = collect_features_1d(
            agent=self,
            dataset_name='validation',
            fetches=[self.model.score],
            feed_dict={self.model.dropout_keep_prob: 1.0})

        # sample loss/summaries for only the first batch
        (summaries, loss) = self.sample_stats(dataset_name='validation')

        self.validation_retention.add_score(score, self.global_step)
        self.tb_logger_validation.log_scalar('score', score, self.global_step)
        self.tb_logger_validation.log_scalar('best score', self.validation_retention.get_best_score(), self.global_step)
        self.summary_writer_validation.add_summary(summaries, self.global_step)
        self.summary_writer_validation.flush()
        self.log.info('EVALUATION (step={}): loss: {}, score: {}, best score: {}' \
                      .format(self.global_step, loss, score, self.validation_retention.get_best_score()))

    def test_step(self):
        '''Implementing one test step.'''
        self.log.info('start running test within training. global_step={}'.format(self.global_step))
        (score,) = collect_features_1d(
            agent=self,
            dataset_name='test',
            fetches=[self.model.score],
            feed_dict={self.model.dropout_keep_prob: 1.0})

        # sample loss/summaries for only the first batch
        (summaries, loss) = self.sample_stats(dataset_name='test')

        self.test_retention.add_score(score, self.global_step)
        self.tb_logger_test.log_scalar('score', score, self.global_step)
        self.tb_logger_test.log_scalar('best score', self.test_retention.get_best_score(), self.global_step)
        self.summary_writer_test.add_summary(summaries, self.global_step)
        self.summary_writer_test.flush()
        self.log.info('TEST (step={}): loss: {}, score: {}, best score: {}' \
                      .format(self.global_step, loss, score, self.test_retention.get_best_score()))

    def set_params(self):
        super(DMLClassificationTrainer, self).set_params()
        dml_margin_multiplier = self.plain_sess.run([self.model.dml_margin_multiplier])

        assign_ops = []
        if not np.isclose(dml_margin_multiplier, self.prm.network.optimization.DML_MARGIN_MULTIPLIER):
            assign_ops.append(self.model.assign_ops['dml_margin_multiplier'])
            self.log.warning('changing model.dml_margin_multiplier from {} to {}'.
                             format(dml_margin_multiplier, self.prm.network.optimization.DML_MARGIN_MULTIPLIER))

        self.plain_sess.run(assign_ops)

    def get_train_summaries(self):
        super(DMLClassificationTrainer, self).get_train_summaries()
        tf.add_to_collection(TRAIN_SUMMARIES, tf.summary.scalar('dml_margin_multiplier', self.model.dml_margin_multiplier))
