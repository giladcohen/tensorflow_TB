'''
In this script I implement the best buddies method for the 'fc7' features,
just before the classifier.
Only support individual nets.
'''

import numpy as np
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
from keras.datasets import cifar10, cifar100 #for debug
from math import fabs, ceil
import multiprocessing as mp
import sharedmem
import sys
import my_lib

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log_root', '', 'log dir of the net to use')
flags.DEFINE_string('network', 'gap', 'mp or gap')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
flags.DEFINE_integer('poolsize', '18', 'number of CPU cores to use in parallel') #for debug
flags.DEFINE_integer('FMAPS', '-1', 'number of features to calculate BBP for') #for debug

DATA_PATH = '/data/gilad/Resnet_KNN/resnet_dump'
sys.stdout = my_lib.Logger()
logs_vec = [FLAGS.log_root] #TODO: support ensembles
N_logs = len(logs_vec)

if (FLAGS.dataset == 'cifar10'):
    NUM_CLASSES = 10
elif(FLAGS.dataset == 'cifar100'):
    NUM_CLASSES = 100

if (FLAGS.FMAPS != -1):
    FMAPS = FLAGS.FMAPS
elif (FLAGS.network == 'gap'):
    FMAPS = 640
elif (FLAGS.network == 'mp'):
    FMAPS = 32

train_images=[]
train_labels=[]
train_logits=[]
train_predictions=[]
train_fc1=[]
train_labels_reshaped=[]
train_predictions_reshaped=[]

test_images=[]
test_labels=[]
test_logits=[]
test_predictions=[]
test_fc1=[]
test_labels_reshaped=[]
test_predictions_reshaped=[]

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

#debug
if   (FLAGS.dataset == 'cifar10'):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
elif (FLAGS.dataset == 'cifar100'):
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode='fine')

#reshaping predictions and labels to (10000,1) arrays instead of (10000,) arrays
N_train = train_labels[0].shape[0]
N_test  = test_labels[0].shape[0]
for i in range(N_logs):
    train_labels_reshaped.append(train_labels[i].reshape(N_train, 1).astype(np.int))
    train_predictions_reshaped.append(train_predictions[i].reshape(N_train, 1).astype(np.int))
    test_labels_reshaped.append(test_labels[i].reshape(N_test,    1).astype(np.int))
    test_predictions_reshaped.append(test_predictions[i].reshape(N_test,    1).astype(np.int))

#selecting train/test data
train_data = train_fc1[0]
test_data  = test_fc1[0]
BBP = np.zeros([N_test, N_train], dtype=np.int)

train_data_shared = sharedmem.copy(train_data)
test_data_shared  = sharedmem.copy(test_data)
BBP_shared        = sharedmem.copy(BBP)

def err_report(label_est, true_labels, label_vote=None):
    '''Function to collect ties and errors'''
    tie_indices=[] #TODO(implement list for all k's)
    err_indices=[]
    for ind in range(N_test):
        if ((not (label_vote is None)) and len(np.nonzero(label_vote[ind]==max(label_vote[ind]))[0])>1):
            #print("There is a tie for test sample #%0d" %ind)
            tie_indices.append(ind)
        if (label_est[ind] != true_labels[ind]):
            #print("There is an error for test sample #%0d" %ind)
            err_indices.append(ind)
    return tie_indices, err_indices

def updateBBP(train_data, test_data ,f ,BBP):
    #print ('calculating best buddy pairs for feature=%0d' %(f))
    pairs_found = 0
    N_train   = train_data.shape[0]
    N_test    = test_data.shape[0]
    D = np.zeros([N_test, N_train], dtype=np.float32) # this matrix is calculated 640 times.
    it = np.nditer(D, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
        it[0] = fabs(test_data[it.multi_index[0], f] - train_data[it.multi_index[1], f])
        it.iternext()
#   for row in xrange(N_test):
#       for col in xrange(N_train):
#           D[row,col] = fabs(test_data[0][row,f] - train_data[0][col,f])
    bb_test  = np.argmin(D, axis=1) #best buddy of every test  sample from the train space
    bb_train = np.argmin(D, axis=0) #best buddy of every train sample from the test  space
    for i in xrange(N_test):
        j = bb_test[i]
        if (bb_train[j] == i):
            #print ('found best buddies! test idx=%0d, train idx=%0d, feature=%0d' %(i, j, f))
            pairs_found += 1
            BBP[i,j]    += 1
    print ('found %0d best buddy pairs for feature=%0d' %(pairs_found, f))

class Counter(object):
    def __init__(self, initval=0):
        self.val = mp.Value('i', initval)
        self.lock = mp.Lock()
    def increment(self):
        with self.lock:
            self.val.value += 1
    def value(self):
        with self.lock:
            return self.val.value
    def value_and_incr(self):
        with self.lock:
            val = self.val.value
            self.val.value += 1
            return val
    def incr_and_value(self):
        with self.lock:
            self.val.value += 1
            return self.val.value


def updateBBP_thread(train_data, test_data ,BBP, counter, core):
    print ('Starting thread %0d.' %(core))
    f = counter.value_and_incr()
    while (f < FMAPS):
        print ('Start calculating best buddy pairs for feature=%0d in thread=%0d' %(f, core))
        updateBBP(train_data, test_data ,f ,BBP)
        print ('finished calculating best buddy pairs for feature=%0d in thread=%0d' %(f, core))
        f = counter.value_and_incr()
    print ('Thread %0d is finished.' %(core))
    
# setting multiprocess variables
poolsize = FLAGS.poolsize
counter = Counter(0)

jobs = []
for core in range(poolsize):
    process = mp.Process(target=updateBBP_thread, 
                         args=(train_data_shared, test_data_shared, BBP_shared, counter, core))
    jobs.append(process)

for j in jobs: j.start()
for j in jobs: j.join()

print('counter.value = %0d' %counter.value())

#save BBP file
BBP[:] = BBP_shared
BBP_file  = os.path.join(DATA_PATH, 'BBP'+FLAGS.log_root[-12:]+'.npy')
np.save(BBP_file, BBP)