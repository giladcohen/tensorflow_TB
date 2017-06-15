
"""ResNet Train/Eval module. In this code I iterate over the entire train or test set - only once,
For a pretrained TF checkpoint
"""
import time
import six
import sys

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

BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
LOG_PATH  = os.path.join(BASE_PATH, FLAGS.log_root)
if (FLAGS.segment == '*'):
    TRAIN_SET_SIZE = 50000
else:
    TRAIN_SET_SIZE = 10000
TEST_SET_SIZE  = 10000
BATCH_SIZE=100
TRAIN_BATCH_COUNT  = TRAIN_SET_SIZE/BATCH_SIZE
TEST_BATCH_COUNT   = TEST_SET_SIZE/BATCH_SIZE

if (FLAGS.eval_data == 'train'):
    SET_SIZE           = TRAIN_SET_SIZE
    BATCH_COUNT        = TRAIN_BATCH_COUNT
    file_pattern       = 'cifar10/data_batch_' + FLAGS.segment + '.bin'
elif (FLAGS.eval_data == 'test'):
    SET_SIZE           = TEST_SET_SIZE
    BATCH_COUNT        = TEST_BATCH_COUNT
    file_pattern       = 'cifar10/test_batch.bin'

if (FLAGS.segment == '*'):
    suffix = ''
else:
    suffix = '_' + FLAGS.segment
print('file_pattern to draw data from: %0s. suffix = %0s' %(file_pattern, suffix))

images_raw_file    = os.path.join(BASE_PATH, FLAGS.save_dir, FLAGS.eval_data+'_images_raw'+suffix+'.npy') # for debug
images_file        = os.path.join(BASE_PATH, FLAGS.save_dir, FLAGS.eval_data+'_images'+suffix+'.npy')     
labels_file        = os.path.join(BASE_PATH, FLAGS.save_dir, FLAGS.eval_data+'_labels'+suffix+'.npy')
init_conv_file     = os.path.join(BASE_PATH, FLAGS.save_dir, FLAGS.eval_data+'_init_conv'+suffix+'.npy')
unit_1_3_conv_file = os.path.join(BASE_PATH, FLAGS.save_dir, FLAGS.eval_data+'_unit_1_3_conv'+suffix+'.npy')
unit_2_3_conv_file = os.path.join(BASE_PATH, FLAGS.save_dir, FLAGS.eval_data+'_unit_2_3_conv'+suffix+'.npy')
unit_3_3_conv_file = os.path.join(BASE_PATH, FLAGS.save_dir, FLAGS.eval_data+'_unit_3_3_conv'+suffix+'.npy')
pool_file          = os.path.join(BASE_PATH, FLAGS.save_dir, FLAGS.eval_data+'_pool'+suffix+'.npy')

hps = resnet_model.HParams(batch_size=BATCH_SIZE,
                             num_classes=10,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=4,
                             use_bottleneck=False,
                             weight_decay_rate=0.0005,
                             relu_leakiness=0.1,
                             optimizer='mom',
                             use_nesterov=True)

images_raw_vec    =        np.ones((SET_SIZE,32,32,3), dtype=np.uint8)
images_vec        = -1.0 * np.ones((SET_SIZE,32,32,3))
labels_vec        = -1.0 * np.ones(SET_SIZE, dtype=np.int)
init_conv_vec     = -1.0 * np.ones((SET_SIZE, 32, 32, 16),  dtype=np.float32)
unit_1_3_conv_vec = -1.0 * np.ones((SET_SIZE, 32, 32, 160), dtype=np.float32)
unit_2_3_conv_vec = -1.0 * np.ones((SET_SIZE, 16, 16, 320), dtype=np.float32)
unit_3_3_conv_vec = -1.0 * np.ones((SET_SIZE,  8,  8, 640), dtype=np.float32)
pool_vec          = -1.0 * np.ones((SET_SIZE, 640), dtype=np.float32)

# net_data = { TODO: support one dictionary
#     'init_conv'    : -1 * np.ones((SET_SIZE, 32, 32, 16),  dtype=np.float32),
#     'unit_1_0_conv': -1 * np.ones((SET_SIZE, 32, 32, 160), dtype=np.float32),
#     'unit_1_1_conv': -1 * np.ones((SET_SIZE, 32, 32, 160), dtype=np.float32),
#     'unit_1_2_conv': -1 * np.ones((SET_SIZE, 32, 32, 160), dtype=np.float32),
#     'unit_1_3_conv': -1 * np.ones((SET_SIZE, 32, 32, 160), dtype=np.float32),
#     'unit_2_0_conv': -1 * np.ones((SET_SIZE, 16, 16, 320), dtype=np.float32),
#     'unit_2_1_conv': -1 * np.ones((SET_SIZE, 16, 16, 320), dtype=np.float32),
#     'unit_2_2_conv': -1 * np.ones((SET_SIZE, 16, 16, 320), dtype=np.float32),
#     'unit_2_3_conv': -1 * np.ones((SET_SIZE, 16, 16, 320), dtype=np.float32),
#     'unit_3_0_conv': -1 * np.ones((SET_SIZE,  8,  8, 640), dtype=np.float32),
#     'unit_3_1_conv': -1 * np.ones((SET_SIZE,  8,  8, 640), dtype=np.float32),
#     'unit_3_2_conv': -1 * np.ones((SET_SIZE,  8,  8, 640), dtype=np.float32),
#     'unit_3_3_conv': -1 * np.ones((SET_SIZE,  8,  8, 640), dtype=np.float32)
# }

images_raw, images, labels = cifar_input.build_input('cifar10',
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
    (summaries, loss, imgs_raw, imgs, net, predictions, truth, train_step) = sess.run(
        [model.summaries, model.cost, images_raw, model._images,
         model.net, model.predictions, model.labels, model.global_step])
    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]
    tmp_prediction = 1.0 * correct_prediction / total_prediction
    b = i*BATCH_SIZE
    e = (i+1)*BATCH_SIZE
    images_raw_vec[b:e]    = imgs_raw
    images_vec[b:e]        = imgs
    labels_vec[b:e]        = truth
    init_conv_vec[b:e]     = net['init_conv']
    unit_1_3_conv_vec[b:e] = net['unit_1_3_conv']
    unit_2_3_conv_vec[b:e] = net['unit_2_3_conv']
    unit_3_3_conv_vec[b:e] = net['unit_3_3_conv']
    pool_vec[b:e]          = np.reshape(net['pool_out'], (BATCH_SIZE, -1))
    print("run %0s batch %0d out of %0d. tmp_prediction = %.4f" %(FLAGS.eval_data, i+1, BATCH_COUNT, tmp_prediction))
 
precision = 1.0 * correct_prediction / total_prediction
print('For %0s set: loss: %.3f, precision: %.3f' % (FLAGS.eval_data, loss, precision))
tf.logging.info('For %0s set: loss: %.3f, precision: %.3f' % (FLAGS.eval_data, loss, precision))

print('Dumping to disc...')
np.save(images_raw_file,    images_raw_vec)
np.save(images_file,        images_vec)
np.save(labels_file,        labels_vec)
np.save(init_conv_file,     init_conv_vec)
np.save(unit_1_3_conv_file, unit_1_3_conv_vec)
np.save(unit_2_3_conv_file, unit_2_3_conv_vec)
np.save(unit_3_3_conv_file, unit_3_3_conv_vec)
np.save(pool_file,          pool_vec)
