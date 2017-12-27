from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.classification_trainer import ClassificationTrainer
from utils.misc import get_vars, get_plain_session, collect_features_1d
import tensorflow as tf

class DMLClassificationTrainer(ClassificationTrainer):
    """Implementing classification trainer for DML"""

    def __init__(self, *args, **kwargs):
        super(DMLClassificationTrainer, self).__init__(*args, **kwargs)
        self.checkpoint_ref = self.prm.train.train_control.CHECKPOINT_REF

    def print_stats(self):
        super(DMLClassificationTrainer, self).print_stats()
        self.log.info(' CHECKPOINT_REF: {}'.format(self.checkpoint_ref))

    def finalize_graph(self):
        # optionally load checkpoint reference
        if self.checkpoint_ref is not None:
            self.log.info('loading pretrained checkpoint file from ref: {}'.format(self.checkpoint_ref))
            vars_to_ignore, vars_to_load = get_vars('RMSProp', 'dml_margin_multiplier', 'fully_connected')
            init_saver = tf.train.Saver(var_list=vars_to_load, name='init_saver', filename='model_ref')
            init_saver.restore(get_plain_session(self.sess), self.checkpoint_ref)
        self.global_step = self.sess.run(self.model.global_step, feed_dict=self._get_dummy_feed())

    def eval_step(self):
        '''Implementing one evaluation step.'''
        self.log.info('start running eval within training. global_step={}'.format(self.global_step))
        (score,) = collect_features_1d(
            agent=self,
            dataset_type='validation',
            fetches=[self.model.score],
            feed_dict={self.model.dropout_keep_prob: 1.0})

        # sample loss/summaries for only the first batch
        (summaries, loss) = self.sample_eval_stats()

        self.validation_retention.add_score(score, self.global_step)
        self.tb_logger_eval.log_scalar('score', score, self.global_step)
        self.tb_logger_eval.log_scalar('best score', self.validation_retention.get_best_score(), self.global_step)
        self.summary_writer_eval.add_summary(summaries, self.global_step)
        self.summary_writer_eval.flush()
        self.log.info('EVALUATION (step={}): loss: {}, score: {}, best score: {}' \
                      .format(self.global_step, loss, score, self.validation_retention.get_best_score()))
