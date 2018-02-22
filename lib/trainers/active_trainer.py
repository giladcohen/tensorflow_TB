from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.trainers.classification_trainer import ClassificationTrainer
from sklearn.decomposition import PCA
import tensorflow as tf

class ActiveTrainer(ClassificationTrainer):
    """Implementing active trainer
    Increasing the labeled pool gradually by using K-Means and K-NN
    Should run with DecayByScoreSetter
    """

    def __init__(self, *args, **kwargs):
        super(ActiveTrainer, self).__init__(*args, **kwargs)
        self.min_learning_rate          = self.prm.train.train_control.MIN_LEARNING_RATE
        self.annotation_rule            = self.prm.train.train_control.ANNOTATION_RULE
        self.steps_for_new_annotations  = self.prm.train.train_control.STEPS_FOR_NEW_ANNOTATIONS
        self.init_after_annot           = self.prm.train.train_control.INIT_AFTER_ANNOT
        self.active_selection_criterion = self.prm.train.train_control.ACTIVE_SELECTION_CRITERION

        self.pca_reduction = self.prm.train.train_control.PCA_REDUCTION
        self.pca_embedding_dims = self.prm.train.train_control.PCA_EMBEDDING_DIMS
        self.pca = PCA(n_components=self.pca_embedding_dims, random_state=self.rand_gen)

        self._activate_annot = True
        self.steps_for_new_annotations = self.steps_for_new_annotations or []
        self.select_new_samples = self.Factories.get_active_selection_fn()

        self._finalized_once = False

    def train(self):
        while not self.sess.should_stop():
            if self.to_annotate():
                self.annot_step()
                self.update_graph()
                self._activate_annot = False
            elif self.to_eval():
                self.eval_step()
                self._activate_eval  = False
            elif self.to_test():
                self.test_step()
                self._activate_test = False
            else:
                self.train_step()
                self._activate_annot = True
                self._activate_eval  = True
                self._activate_test  = True
        self.log.info('Stop training at global_step={}'.format(self.global_step))

    def annot_step(self):
        '''Implementing one annotation step'''
        self.log.info('Adding {} new labels to train dataset.'.format(self.dataset.clusters))
        new_indices = self.select_new_samples(self)  # select new indices
        self.dataset.update_pool(indices=new_indices)           # add new indices to train dataset

        # reset learning rate to initial value, retention memory and model weights
        if self.init_after_annot:
            self.init_weights()
        self.learning_rate_hook.reset_learning_rate()
        self.validation_retention.reset_memory()

    def update_graph(self):
        """Resetting the graph and starting a new graph to update the dataset operations on the graph"""
        tf.reset_default_graph()
        train_validation_map_ref = self.dataset.train_validation_map_ref

        self.model   = self.Factories.get_model()
        self.dataset = self.Factories.get_dataset()
        self.dataset.train_validation_map_ref = train_validation_map_ref
        self.build()
        self.log.info('Done restoring graph for global_step ({})'.format(self.global_step))

    def finalize_graph(self):
        """overwrite the global step on the graph"""
        if self._finalized_once:
            self.log.info('overwriting graph\'s value: global_step={}'.format(self.global_step))
            self.plain_sess.run(self.model.assign_ops['global_step_ow'],
                                feed_dict={self.model.global_step_ph: self.global_step})
            self.dataset.set_handles(self.plain_sess)
        else:
            # restoring global_step
            super(ActiveTrainer, self).finalize_graph()
            self._finalized_once = True

    def print_stats(self):
        super(ActiveTrainer, self).print_stats()
        self.log.info(' MIN_LEARNING_RATE: {}'.format(self.min_learning_rate))
        self.log.info(' PCA_REDUCTION: {}'.format(self.pca_reduction))
        self.log.info(' PCA_EMBEDDING_DIMS: {}'.format(self.pca_embedding_dims))
        self.log.info(' ANNOTATION_RULE: {}'.format(self.annotation_rule))
        self.log.info(' STEPS_FOR_NEW_ANNOTATIONS: {}'.format(self.steps_for_new_annotations))
        self.log.info(' INIT_AFTER_ANNOT: {}'.format(self.init_after_annot))

    def to_annotate(self):
        """
        :return: boolean. Whether or not to start an annotation phase
        """

        if not self._activate_annot:
            return False

        if self.annotation_rule == 'small_learning_rate':
            ret = self.learning_rate_hook.get_lrn_rate() < self.min_learning_rate and self.dataset.pool_size < self.dataset.cap
        elif self.annotation_rule == 'fixed_epochs':
            ret = self.global_step in self.steps_for_new_annotations
        else:
            err_str = 'annotation_rule={} is not supported'.format(self.annotation_rule)
            self.log.error(err_str)
            raise AssertionError(err_str)
        return ret

    def init_weights(self):
        self.log.info('Start initializing weights in global step={}'.format(self.global_step))
        self.plain_sess.run(self.model.init_op)
        self.log.info('Done initializing weights in global step={}'.format(self.global_step))

        # restore model global_step
        self.plain_sess.run(self.model.assign_ops['global_step_ow'],
                            feed_dict={self.model.global_step_ph: self.global_step})
        self.log.info('Done restoring global_step ({})'.format(self.global_step))

    def train_step(self):
        '''Implementing one training step'''
        _, images, labels = self.dataset.get_mini_batch('train_pool', self.plain_sess)
        _ , self.global_step = self.sess.run([self.model.train_op, self.model.global_step],
                                              feed_dict={self.model.images: images,
                                                         self.model.labels: labels,
                                                         self.model.is_training: True})
