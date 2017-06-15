'''
Here I load BPP matrix and estimate a label for every test sample
'''

import numpy as np
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
from keras.datasets import cifar10, cifar100 #for debug

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log_root', '', 'log dir of the net to use')
flags.DEFINE_string('network', 'gap', 'mp or gap')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
flags.DEFINE_string('mode', 'plain', 'mode of analysis')

DATA_PATH = '/data/gilad/Resnet_KNN/resnet_dump'

if (FLAGS.log_root == 'cifar10_gap_ensemble'):
    logs_vec = ['logs_wrn28-10_1513_220317', \
                'logs_wrn28-10_0137_050417', \
                'logs_wrn28-10_0203_090417', \
                'logs_wrn28-10_0853_120417', \
                'logs_wrn28-10_0732_140417', \
                'logs_wrn28-10_1434_160417', \
                'logs_wrn28-10_1753_170417']
elif (FLAGS.log_root == 'cifar100_gap_ensemble'):
    logs_vec = ['logs_wrn28-10_2101_110517', \
                'logs_wrn28-10_2102_110517', \
                'logs_wrn28-10_2103_110517', \
                'logs_wrn28-10_1944_120517', \
                'logs_wrn28-10_1945_120517', \
                'logs_wrn28-10_1946_120517', \
                'logs_wrn28-10_1546_140517', \
                'logs_wrn28-10_1547_140517', \
                'logs_wrn28-10_1548_140517',]
elif (FLAGS.log_root != ''):
    logs_vec = [FLAGS.log_root]
else:
    print ('please enter log_root')
    
N_logs = len(logs_vec)

if (FLAGS.dataset == 'cifar10'):
    NUM_CLASSES = 10
elif(FLAGS.dataset == 'cifar100'):
    NUM_CLASSES = 100

if (FLAGS.network == 'gap'):
    FMAPS = 640
elif (FLAGS.network == 'mp'):
    FMAPS = 32

train_images=[]
train_labels=[]
train_logits=[]
train_predictions=[]
train_fc1=[]
train_predictions_reshaped=[]

test_images=[]
test_labels=[]
test_logits=[]
test_predictions=[]
test_fc1=[]
test_predictions_reshaped=[]

BBP_vec = []

#loading data
for i in range(N_logs):
    print('reading data for i=%0d' %i)
    suffix = logs_vec[i][-12:]
    train_images.append(np.load(os.path.join(DATA_PATH, 'train_images_raw'+suffix+'.npy')))
    train_labels.append(np.load(os.path.join(DATA_PATH, 'train_labels'+suffix+'.npy')))
    train_logits.append(np.load(os.path.join(DATA_PATH, 'train_logits'+suffix+'.npy')))
    train_predictions.append(np.load(os.path.join(DATA_PATH, 'train_predictions'+suffix+'.npy')))
    train_fc1.append(np.load(os.path.join(DATA_PATH,    'train_fc1'+suffix+'.npy')))
    test_images.append(np.load(os.path.join(DATA_PATH,  'test_images_raw'+suffix+'.npy')))
    test_labels.append(np.load(os.path.join(DATA_PATH,  'test_labels'+suffix+'.npy')))
    test_logits.append(np.load(os.path.join(DATA_PATH,  'test_logits'+suffix+'.npy')))
    test_predictions.append(np.load(os.path.join(DATA_PATH, 'test_predictions'+suffix+'.npy')))
    test_fc1.append(np.load(os.path.join(DATA_PATH,     'test_fc1'+suffix+'.npy')))
    BBP_vec.append(np.load(os.path.join(DATA_PATH,      'BBP'+suffix+'.npy')))

#assertions
for i in range(1,N_logs):
    print('asserting i=%0d for images/labels' %i)
    assert (train_images[i]   == train_images[0]).all()
    assert (train_labels[i]   == train_labels[0]).all()
    assert (test_images[i]    == test_images[0]).all()
    assert (test_labels[i]    == test_labels[0]).all()
for i in range(N_logs):
    print('asserting i=%0d for recorded data' %i)
    assert np.sum(train_logits[i]   == -1) == 0
    assert np.sum(train_fc1[i]      == -1) == 0
    assert np.sum(test_logits[i]    == -1) == 0
    assert np.sum(test_fc1[i]       == -1) == 0

#debug
if (FLAGS.dataset == 'cifar10'):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
elif (FLAGS.dataset == 'cifar100'):
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='fine')

#reshaping vectors, defining useable data
train_labels = train_labels[0].astype(np.int)
test_labels  = test_labels[0].astype(np.int)
N_train = train_labels.shape[0]
N_test  = test_labels.shape[0]
train_labels_reshaped = train_labels.reshape(N_train, 1).astype(np.int)
test_labels_reshaped  = test_labels.reshape(N_test,   1).astype(np.int)


BBP = np.zeros([N_test, N_train], dtype=np.int)
for i in range(N_logs):
    train_predictions_reshaped.append(train_predictions[i].reshape(N_train, 1).astype(np.int))
    test_predictions_reshaped.append(test_predictions[i].reshape(N_test,    1).astype(np.int))
    BBP += BBP_vec[i]

#calculating accuracy
if (FLAGS.mode == 'plain'):
    closest_train_sample = np.argmax(BBP, axis=1)
    label_est = train_labels[closest_train_sample]
elif (FLAGS.mode == 'colabel'):
    label_indices = np.empty([NUM_CLASSES, N_train/NUM_CLASSES], dtype=np.int)
    BBP_colabel   = np.empty([N_test, NUM_CLASSES], dtype = np.int)
    for i in xrange(NUM_CLASSES):
        label_indices[i] = np.where(train_labels == i)[0]
        BBP_colabel[:,i] = np.sum(BBP[:,label_indices[i]], axis=1)
    label_est = np.argmax(BBP_colabel, axis=1)
accuracy = np.average(label_est == test_labels)
print('Accuracy for mode %0s: %.4f' %(FLAGS.mode, accuracy))   




 