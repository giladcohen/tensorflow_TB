from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.classification_trainer import ClassificationTrainer
from utils.misc import get_vars, get_plain_session, collect_features_1d
import tensorflow as tf
import os

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
                                                    'fully_connected',
                                                    'unit_3_3',
                                                    'unit_last')
            init_saver = tf.train.Saver(var_list=vars_to_load, name='init_saver', filename='model_ref')
            init_saver.restore(sess, self.checkpoint_ref)
            self.log.info('writing graph with all variables to current checkpoint dir {}'.format(self.checkpoint_dir))
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            self.saver.save(sess, os.path.join(self.checkpoint_dir, 'model.ckpt'))

    def eval_step(self):
        '''Implementing one evaluation step.'''
        self.log.info('start running eval within training. global_step={}'.format(self.global_step))
        (score,) = collect_features_1d(
            agent=self,
            dataset_type='validation',
            fetches=[self.model.score],
            feed_dict={self.model.dropout_keep_prob: 1.0},
            num_samples=self.num_eval_samples)

        # sample loss/summaries for only the first batch
        (summaries, loss) = self.sample_eval_stats()

        self.validation_retention.add_score(score, self.global_step)
        self.tb_logger_eval.log_scalar('score', score, self.global_step)
        self.tb_logger_eval.log_scalar('best score', self.validation_retention.get_best_score(), self.global_step)
        self.summary_writer_eval.add_summary(summaries, self.global_step)
        self.summary_writer_eval.flush()
        self.log.info('EVALUATION (step={}): loss: {}, score: {}, best score: {}' \
                      .format(self.global_step, loss, score, self.validation_retention.get_best_score()))
