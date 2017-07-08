
"""ResNet Train/Eval module.
"""
import time
import six
import sys
import os
dirname=os.path.dirname
cwd = os.getcwd()        #Resnet_KNN
pardir = dirname(cwd)    #workspace
sys.path.insert(0, cwd)
print ('cwd = %s' % cwd)
import numpy as np
import lib.resnet_model as resnet_model
import tensorflow as tf
from keras.datasets import cifar10, cifar100 # for debug
import matplotlib.pyplot as plt #for debug
import lib.active_kmean as active_kmean
from lib.data_tank import DataTank
from lib.active_data_tank import ActiveDataTank
from sklearn.neighbors import NearestNeighbors
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
flags.DEFINE_string('mode', 'train', 'train or eval.')
flags.DEFINE_string('train_data_dir',    os.path.join(pardir, 'cifar10_data/train_data'),       'dir for training data.')
flags.DEFINE_string('train_labels_file', os.path.join(pardir, 'cifar10_data/train_labels.txt'), 'file for training labels.')
flags.DEFINE_string('eval_data_dir',     os.path.join(pardir, 'cifar10_data/test_data'),        'dir for evaluation data.')
flags.DEFINE_string('eval_labels_file',  os.path.join(pardir, 'cifar10_data/test_labels.txt'),  'file for evaluation labels.')
flags.DEFINE_integer('image_size', 32, 'Image side length.')
flags.DEFINE_integer('eval_batch_count', 50, 'Number of batches to eval.')
flags.DEFINE_bool('eval_once', False, 'Whether evaluate the model only once.')
flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used for training. (0 or 1)')
flags.DEFINE_bool('active_learning', False, 'Use active learning')
flags.DEFINE_integer('batch_size', -1, 'batch size for train/test')
flags.DEFINE_integer('clusters', 100, 'batch size for train/test')
flags.DEFINE_integer('cap', 1000, 'batch size for train/test')
flags.DEFINE_integer('active_epochs', 5, 'number of epochs for every pool iteration')


TRAIN_DIR = os.path.join(FLAGS.log_root, 'train')
EVAL_DIR  = os.path.join(FLAGS.log_root, 'test')
TRAIN_SET_SIZE = 50000
TEST_SET_SIZE  = 10000
ACTIVE_EPOCHS  = FLAGS.active_epochs

if (FLAGS.dataset   == 'cifar10'):
    NUM_CLASSES = 10
elif (FLAGS.dataset == 'cifar100'):
    NUM_CLASSES = 100
else:
    raise NameError('Test does not support %s dataset' %FLAGS.dataset)

if (FLAGS.batch_size != -1):
    BATCH_SIZE = FLAGS.batch_size
elif (FLAGS.mode == 'train'):
  BATCH_SIZE=100
else:
  BATCH_SIZE=100

def train(hps):
    """Training loop."""
    """Training loop. Step1 - selecting TRAIN_BATCH randomized images"""
    dt = ActiveDataTank(n_clusters=FLAGS.clusters,
                        data_path=FLAGS.train_data_dir,
                        label_file=FLAGS.train_labels_file,
                        batch_size=BATCH_SIZE,
                        N=TRAIN_SET_SIZE,
                        to_preprocess=True)

    images_ph = tf.placeholder(tf.float32, [BATCH_SIZE, FLAGS.image_size, FLAGS.image_size, 3])
    labels_ph = tf.placeholder(tf.int32,   [BATCH_SIZE])

    images_norm = tf.map_fn(tf.image.per_image_standardization, images_ph)
    labels_1hot = tf.one_hot(labels_ph, NUM_CLASSES)

    model = resnet_model.ResNet(hps, images_norm, labels_1hot, FLAGS.mode)
    model.build_graph()

    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    images_summary = tf.summary.image('images', images_ph, max_outputs=3)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=1, #was 100
        #save_secs = 60,
        output_dir=TRAIN_DIR,
        summary_op=tf.summary.merge([model.summaries,
                                     images_summary,
                                     tf.summary.scalar('Precision', precision)]))

    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss_xent': model.xent_cost,
                 'loss_wd': model.wd_cost,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=1) #was 100

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """Sets learning_rate based on global step."""

        def begin(self):
            self._lrn_rate = FLAGS.learning_rate

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                model.global_step,  # Asks for global step value.
                feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

        def after_run(self, run_context, run_values):
            train_step = run_values.results
            epoch = (BATCH_SIZE*train_step) // FLAGS.cap #was TRAIN_SET_SIZE
            if epoch < 60:
                self._lrn_rate = FLAGS.learning_rate
            elif epoch < 120:
                self._lrn_rate = FLAGS.learning_rate/5
            elif epoch < 160:
                self._lrn_rate = FLAGS.learning_rate/25
            else:
                self._lrn_rate = FLAGS.learning_rate/125

    sess = tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.log_root,
            hooks=[logging_hook, _LearningRateSetterHook()],
            chief_only_hooks=[summary_hook],
            # Since we provide a SummarySaverHook, we need to disable default
            # SummarySaverHook. To do that we set save_summaries_steps to 0.
            save_summaries_steps=0,
            save_checkpoint_secs=600, # was 600
            config=tf.ConfigProto(allow_soft_placement=True))

    if (FLAGS.active_learning == False):
        while not sess.should_stop():
            lp = len(dt.pool)
            if lp == FLAGS.cap:
                images, labels, images_aug, labels_aug = dt.fetch_batch()
                sess.run(model.train_op, feed_dict={images_ph: images_aug,
                                                    labels_ph: labels_aug})
            else:
                steps_to_go = ACTIVE_EPOCHS * (lp / BATCH_SIZE)
                for i in range(steps_to_go):
                    images, labels, images_aug, labels_aug = dt.fetch_batch()
                    sess.run(model.train_op, feed_dict={images_ph: images_aug,
                                                        labels_ph: labels_aug})
                dt.update_pool_rand()
    else:
        while not sess.should_stop():
            lp = len(dt.pool)
            if lp == FLAGS.cap:
                images, labels, images_aug, labels_aug = dt.fetch_batch()
                sess.run(model.train_op, feed_dict={images_ph: images_aug,
                                                    labels_ph: labels_aug})
            else:
                steps_to_go = ACTIVE_EPOCHS * (lp / BATCH_SIZE)
                for i in range(steps_to_go):
                    images, labels, images_aug, labels_aug = dt.fetch_batch()
                    sess.run(model.train_op, feed_dict={images_ph: images_aug,
                                                        labels_ph: labels_aug})

                # analyzing (evaluation)
                model.mode = 'eval'
                fc1_vec = -1.0 * np.ones((TRAIN_SET_SIZE, 640), dtype=np.float32)
                batches_to_store = TRAIN_SET_SIZE / BATCH_SIZE
                print ('start storing feature maps for the entire train set')
                for i in range(batches_to_store):
                    b = i * BATCH_SIZE
                    e = (i + 1) * BATCH_SIZE
                    images, labels, _, _ = dt.fetch_batch_common(indices=range(b,e))
                    net = sess.run(model.net, feed_dict={images_ph: images,
                                                        labels_ph:  labels})
                    fc1_vec[b:e] = np.reshape(net['pool_out'], (BATCH_SIZE, 640))
                    print ('Storing completed: %0d%%' %(int(100.0*float(i)/batches_to_store)))
                assert np.sum(fc1_vec == -1) == 0 #debug
                KM = active_kmean.KMeansWrapper(fixed_centers=fc1_vec[dt.pool], n_clusters=lp + FLAGS.clusters, init='k-means++', n_init=1,
                                                max_iter=300, tol=1e-4, precompute_distances='auto',
                                                verbose=0, random_state=None, copy_x=True,
                                                n_jobs=1, algorithm='auto')
                centers = KM.fit_predict_centers(fc1_vec)
                new_centers = centers[lp:(lp + FLAGS.clusters)]
                nbrs = NearestNeighbors(n_neighbors=1)
                nbrs.fit(fc1_vec)
                indices = nbrs.kneighbors(new_centers, return_distance=False)

                indices = indices.T[0].tolist()
                already_pooled_cnt = 0 # number of indices of samples that we added to pool already
                for myItem in indices:
                    if myItem in dt.pool:
                        already_pooled_cnt += 1
                        indices.remove(myItem)
                        print('Removing value %0d from indices because it already exists in pool' % myItem)
                print('%0d indices were already in pool. Randomized indices will be chosen instead of them' % already_pooled_cnt)
                dt.update_pool(indices)
                dt.update_pool_rand(n_clusters=already_pooled_cnt)

                model.mode = 'train'
                #end of analysis

def evaluate(hps):
    """Eval loop."""

    dt = DataTank(data_path=FLAGS.eval_data_dir,
                  label_file=FLAGS.eval_labels_file,
                  batch_size=BATCH_SIZE,
                  N=TEST_SET_SIZE,
                  to_preprocess=False)

    images_ph = tf.placeholder(tf.float32, [BATCH_SIZE, FLAGS.image_size, FLAGS.image_size, 3])
    labels_ph = tf.placeholder(tf.int32, [BATCH_SIZE])

    images_norm = tf.map_fn(tf.image.per_image_standardization, images_ph)
    labels_1hot = tf.one_hot(labels_ph, NUM_CLASSES)

    model = resnet_model.ResNet(hps, images_norm, labels_1hot, FLAGS.mode)
    model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(EVAL_DIR)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    best_precision = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
            continue
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
            continue
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_prediction = 0, 0
        for i in range(FLAGS.eval_batch_count):
            b = i * BATCH_SIZE
            e = (i + 1) * BATCH_SIZE
            images, labels, _, _ = dt.fetch_batch_common(indices=range(b, e))
            (summaries, loss, predictions, truth, train_step) = sess.run(
                [model.summaries, model.cost, model.predictions, model.labels, model.global_step],
                feed_dict={images_ph: images, labels_ph: labels})

            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]

        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        precision_summ = tf.Summary()
        precision_summ.value.add(tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)
        summary_writer.add_summary(summaries, train_step)
        tf.logging.info('loss: %.4f, precision: %.4f, best precision: %.4f' %
                        (loss, precision, best_precision))
        summary_writer.flush()

        if FLAGS.eval_once:
          break

        time.sleep(60)


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    hps = resnet_model.HParams(batch_size=BATCH_SIZE,
                               num_classes=NUM_CLASSES,
                               min_lrn_rate=0.0001,
                               lrn_rate=FLAGS.learning_rate,
                               num_residual_units=4, #was 5 in source code
                               use_bottleneck=False,
                               xent_rate=1.0,
                               weight_decay_rate=0.0005, #was 0.0002
                               relu_leakiness=0.1,
                               pool='gap', #use gap or mp
                               optimizer='mom',
                               use_nesterov=True)

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode == 'eval':
            evaluate(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()