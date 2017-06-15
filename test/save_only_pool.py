
"""ResNet Train/Eval module. In this code I iterate over the entire train or test set - only once,
For a pretrained TF checkpoint and save the pool_in and other signals
Now this code takes real logits, and not the prediction
"""
import six
import cifar_input
import numpy as np
import resnet_model
import tensorflow as tf
import os.path

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('eval_data', 'test', 'train or test data to extract features from.')
flags.DEFINE_string('segment', '*', 'train or test data to extract features from.')
flags.DEFINE_string('save_dir', 'tmp', 'train or test data to extract features from.')
flags.DEFINE_string('log_root', '', 'parent dir of train and test.')
flags.DEFINE_string('network', 'gap', 'gap or mp.')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')


BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
LOG_PATH  = os.path.join(BASE_PATH, FLAGS.log_root)
if (FLAGS.segment == '*'):
    TRAIN_SET_SIZE = 50000
else:
    TRAIN_SET_SIZE = 10000 #for cifar10
TEST_SET_SIZE  = 10000
BATCH_SIZE=100
TRAIN_BATCH_COUNT  = TRAIN_SET_SIZE/BATCH_SIZE
TEST_BATCH_COUNT   = TEST_SET_SIZE/BATCH_SIZE
if (FLAGS.dataset == 'cifar10'):
    NUM_CLASSES = 10
elif (FLAGS.dataset == 'cifar100'):
    NUM_CLASSES = 100

if (FLAGS.eval_data == 'train'):
    SET_SIZE           = TRAIN_SET_SIZE
    BATCH_COUNT        = TRAIN_BATCH_COUNT
    if (FLAGS.dataset == 'cifar10'):
        file_pattern       = 'cifar10/data_batch_' + FLAGS.segment + '.bin'
    elif (FLAGS.dataset == 'cifar100'):
        file_pattern       = 'cifar100/train.bin'
elif (FLAGS.eval_data == 'test'):
    SET_SIZE           = TEST_SET_SIZE
    BATCH_COUNT        = TEST_BATCH_COUNT
    if (FLAGS.dataset == 'cifar10'):
        file_pattern       = 'cifar10/test_batch.bin'
    elif (FLAGS.dataset == 'cifar100'):
        file_pattern   = 'cifar100/test.bin'

if (FLAGS.segment == '*'):
    suffix = ''
else:
    suffix = '_' + FLAGS.segment
print('Running network %s. Dataset is %0s. File_pattern to draw data from: %0s. suffix = %0s' \
      %(FLAGS.network, FLAGS.dataset, file_pattern, suffix))

save_suffix = FLAGS.log_root[-12:] #yield the date of the generated network

images_raw_file    = os.path.join(BASE_PATH, FLAGS.save_dir, \
                                  FLAGS.eval_data+'_images_raw'+save_suffix+suffix+'.npy') # for debug
labels_file        = os.path.join(BASE_PATH, FLAGS.save_dir, \
                                  FLAGS.eval_data+'_labels'+save_suffix+suffix+'.npy')
logits_file        = os.path.join(BASE_PATH, FLAGS.save_dir, \
                                  FLAGS.eval_data+'_logits'+save_suffix+suffix+'.npy')
predictions_file   = os.path.join(BASE_PATH, FLAGS.save_dir, \
                                  FLAGS.eval_data+'_predictions'+save_suffix+suffix+'.npy')
fc1_file           = os.path.join(BASE_PATH, FLAGS.save_dir, \
                                  FLAGS.eval_data+'_fc1'+save_suffix+suffix+'.npy')

hps = resnet_model.HParams(batch_size=BATCH_SIZE,
                             num_classes=NUM_CLASSES,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=4,
                             use_bottleneck=False,
                             weight_decay_rate=0.0005,
                             relu_leakiness=0.1,
                             pool=FLAGS.network,
                             optimizer='mom',
                             use_nesterov=True)

images_raw_vec  =        np.ones((SET_SIZE,32,32,3), dtype=np.uint8)
labels_vec      = -1.0 * np.ones(SET_SIZE, dtype=np.int)
logits_vec      = -1.0 * np.ones((SET_SIZE,NUM_CLASSES), dtype=np.int)
predictions_vec =        np.ones(SET_SIZE, dtype=np.int)
if (FLAGS.network == 'mp'):
    fc1_vec         = -1.0 * np.ones((SET_SIZE, 32), dtype=np.float32)
elif (FLAGS.network == 'gap'):
    fc1_vec         = -1.0 * np.ones((SET_SIZE, 640), dtype=np.float32)

images_raw, images, labels = cifar_input.build_input(FLAGS.dataset,
                    os.path.join(BASE_PATH,file_pattern), BATCH_SIZE, 'eval', shuffle=False)

model = resnet_model.ResNet(hps, images, labels, 'eval')
model.build_graph()
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

tf.train.start_queue_runners(sess)

try:
    ckpt_state = tf.train.get_checkpoint_state(LOG_PATH)
except tf.errors.OutOfRangeError as e:
    tf.logging.error('Cannot restore checkpoint: %s', e)
if not (ckpt_state and ckpt_state.model_checkpoint_path):
    tf.logging.info('No model to eval yet at %s', LOG_PATH)
tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
saver.restore(sess, ckpt_state.model_checkpoint_path)

total_prediction, correct_prediction = 0, 0
for i in six.moves.range(BATCH_COUNT):
    (summaries, loss, imgs_raw, net, predictions_raw, truth, train_step) = sess.run(
        [model.summaries, model.cost, images_raw,
         model.net, model.predictions, model.labels, model.global_step])
    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions_raw, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]
    tmp_prediction = 1.0 * correct_prediction / total_prediction
    b = i*BATCH_SIZE
    e = (i+1)*BATCH_SIZE
    images_raw_vec[b:e]    = imgs_raw
    labels_vec[b:e]        = truth
    logits_vec[b:e]        = predictions_raw
    predictions_vec[b:e]   = predictions
    if (FLAGS.network == 'mp'):
        fc1_vec[b:e]       = np.reshape(net['fc1'], (BATCH_SIZE,32))
    elif (FLAGS.network == 'gap'):
        fc1_vec[b:e]       = np.reshape(net['pool_out'], (BATCH_SIZE,640))
    #print("run %0s batch %0d out of %0d. tmp_prediction = %.4f" %(FLAGS.eval_data, i+1, BATCH_COUNT, tmp_prediction))
 
precision = 1.0 * correct_prediction / total_prediction
print('For %0s set: loss: %.4f, precision: %.4f' % (FLAGS.eval_data, loss, precision))
tf.logging.info('For %0s set: loss: %.4f, precision: %.4f' % (FLAGS.eval_data, loss, precision))

print('Dumping to disc...')
np.save(images_raw_file,    images_raw_vec)
np.save(labels_file,        labels_vec)
np.save(logits_file,        logits_vec)
np.save(predictions_file,   predictions_vec)
np.save(fc1_file,           fc1_vec)
