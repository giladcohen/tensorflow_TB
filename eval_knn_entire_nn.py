'''Loading train and test many feature maps along WRN-28-10 ResNet
And implementing k-NN for different k's for the output of average pooling.
'''
import numpy as np
import os.path
import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10 #for debug
import tensorflow as tf
import my_lib

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('buffer'   , 'pool', 'name of buffer in the network.')
flags.DEFINE_integer('fraction',   100 , 'Fraction of train set to use due to insufficient memory')

BASE_PATH = '/data'
mb   = 10000
M = (FLAGS.fraction*10000) / 100 #total number of samples to account for

#loading train and test data
train_images_raw_1    = np.load(os.path.join(BASE_PATH, 'train_images_raw_1.npy'))
train_images_raw_2    = np.load(os.path.join(BASE_PATH, 'train_images_raw_2.npy'))
train_images_raw_3    = np.load(os.path.join(BASE_PATH, 'train_images_raw_3.npy'))
train_images_raw_4    = np.load(os.path.join(BASE_PATH, 'train_images_raw_4.npy'))
train_images_raw_5    = np.load(os.path.join(BASE_PATH, 'train_images_raw_5.npy'))
test_images_raw       = np.load(os.path.join(BASE_PATH, 'test_images_raw.npy'))

train_images_1        = np.load(os.path.join(BASE_PATH, 'train_images_1.npy'))
train_images_2        = np.load(os.path.join(BASE_PATH, 'train_images_2.npy'))
train_images_3        = np.load(os.path.join(BASE_PATH, 'train_images_3.npy'))
train_images_4        = np.load(os.path.join(BASE_PATH, 'train_images_4.npy'))
train_images_5        = np.load(os.path.join(BASE_PATH, 'train_images_5.npy'))
test_images           = np.load(os.path.join(BASE_PATH, 'test_images.npy'))

train_labels_1       = np.load(os.path.join(BASE_PATH, 'train_labels_1.npy'))
train_labels_2       = np.load(os.path.join(BASE_PATH, 'train_labels_2.npy'))
train_labels_3       = np.load(os.path.join(BASE_PATH, 'train_labels_3.npy'))
train_labels_4       = np.load(os.path.join(BASE_PATH, 'train_labels_4.npy'))
train_labels_5       = np.load(os.path.join(BASE_PATH, 'train_labels_5.npy'))
test_labels          = np.load(os.path.join(BASE_PATH, 'test_labels.npy'))

if (FLAGS.buffer == 'image'):
    train_data_1     = train_images_1.astype(np.float32)
    train_data_2     = train_images_2.astype(np.float32)
    train_data_3     = train_images_3.astype(np.float32)
    train_data_4     = train_images_4.astype(np.float32)
    train_data_5     = train_images_5.astype(np.float32)
    test_data        = test_images.astype(np.float32)
elif (FLAGS.buffer == 'init_conv'):
    train_data_1     = np.load(os.path.join(BASE_PATH, 'train_init_conv_1.npy'))
    train_data_2     = np.load(os.path.join(BASE_PATH, 'train_init_conv_2.npy'))
    train_data_3     = np.load(os.path.join(BASE_PATH, 'train_init_conv_3.npy'))
    train_data_4     = np.load(os.path.join(BASE_PATH, 'train_init_conv_4.npy'))
    train_data_5     = np.load(os.path.join(BASE_PATH, 'train_init_conv_5.npy'))
    test_data        = np.load(os.path.join(BASE_PATH, 'test_init_conv.npy'))
    mb = 100
elif (FLAGS.buffer == 'unit_conv_1_3'):
    train_data          = np.empty([M * 5, 32 , 32, 160], dtype=np.float32)
    test_data           = np.empty([M,     32 , 32, 160], dtype=np.float32)
    p = np.random.permutation(10000)
    train_data[0:M]     = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_1.npy'))[p][0:M]
    train_labels_1 = train_labels_1[p][0:M]
    p = np.random.permutation(10000)
    train_data[M:2*M]   = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_2.npy'))[p][0:M]
    train_labels_2 = train_labels_2[p][0:M]
    p = np.random.permutation(10000)
    train_data[2*M:3*M] = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_3.npy'))[p][0:M]
    train_labels_3 = train_labels_3[p][0:M]
    p = np.random.permutation(10000)
    train_data[3*M:4*M] = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_4.npy'))[p][0:M]
    train_labels_4 = train_labels_4[p][0:M]
    p = np.random.permutation(10000)
    train_data[4*M:5*M] = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_5.npy'))[p][0:M]
    train_labels_5 = train_labels_5[p][0:M]
    p = np.random.permutation(10000)
    test_data           = np.load(os.path.join(BASE_PATH, 'test_unit_1_3_conv.npy'))[p][0:M]
    test_labels         = test_labels[p][0:M]
    #train_data_1     = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_1.npy')) #, mmap_mode='c')
    #train_data_2     = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_2.npy')) #, mmap_mode='c')
    #train_data_3     = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_3.npy')) #, mmap_mode='c')
    #train_data_4     = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_4.npy')) #, mmap_mode='c')
    #train_data_5     = np.load(os.path.join(BASE_PATH, 'train_unit_1_3_conv_5.npy')) #, mmap_mode='c')
    #test_data        = np.load(os.path.join(BASE_PATH, 'test_unit_1_3_conv.npy'))    #, mmap_mode='c')
    mb = 10
elif (FLAGS.buffer == 'unit_conv_2_3'):
    train_data_1     = np.load(os.path.join(BASE_PATH, 'train_unit_2_3_conv_1.npy'))
    train_data_2     = np.load(os.path.join(BASE_PATH, 'train_unit_2_3_conv_2.npy'))
    train_data_3     = np.load(os.path.join(BASE_PATH, 'train_unit_2_3_conv_3.npy'))
    train_data_4     = np.load(os.path.join(BASE_PATH, 'train_unit_2_3_conv_4.npy'))
    train_data_5     = np.load(os.path.join(BASE_PATH, 'train_unit_2_3_conv_5.npy'))
    test_data        = np.load(os.path.join(BASE_PATH, 'test_unit_2_3_conv.npy'))
    mb = 100
elif (FLAGS.buffer == 'unit_conv_3_3'):
    train_data_1     = np.load(os.path.join(BASE_PATH, 'train_unit_3_3_conv_1.npy'))
    train_data_2     = np.load(os.path.join(BASE_PATH, 'train_unit_3_3_conv_2.npy'))
    train_data_3     = np.load(os.path.join(BASE_PATH, 'train_unit_3_3_conv_3.npy'))
    train_data_4     = np.load(os.path.join(BASE_PATH, 'train_unit_3_3_conv_4.npy'))
    train_data_5     = np.load(os.path.join(BASE_PATH, 'train_unit_3_3_conv_5.npy'))
    test_data        = np.load(os.path.join(BASE_PATH, 'test_unit_3_3_conv.npy'))
    mb = 100
elif (FLAGS.buffer == 'pool'):
    train_data_1     = np.load(os.path.join(BASE_PATH, 'train_pool_1.npy'))
    train_data_2     = np.load(os.path.join(BASE_PATH, 'train_pool_2.npy'))
    train_data_3     = np.load(os.path.join(BASE_PATH, 'train_pool_3.npy'))
    train_data_4     = np.load(os.path.join(BASE_PATH, 'train_pool_4.npy'))
    train_data_5     = np.load(os.path.join(BASE_PATH, 'train_pool_5.npy'))
    test_data        = np.load(os.path.join(BASE_PATH, 'test_pool.npy'))

# randomizing only FLAGS.fraction percent of the data for k-NN
if (FLAGS.buffer != 'unit_conv_1_3'):
    p = np.random.permutation(10000)
    train_data_1   = train_data_1[p][0:M]
    train_labels_1 = train_labels_1[p][0:M]
    p = np.random.permutation(10000)
    train_data_2   = train_data_2[p][0:M]
    train_labels_2 = train_labels_2[p][0:M]
    p = np.random.permutation(10000)
    train_data_3   = train_data_3[p][0:M]
    train_labels_3 = train_labels_3[p][0:M]
    p = np.random.permutation(10000)
    train_data_4   = train_data_4[p][0:M]
    train_labels_4 = train_labels_4[p][0:M]
    p = np.random.permutation(10000)
    train_data_5   = train_data_5[p][0:M]
    train_labels_5 = train_labels_5[p][0:M]
    p = np.random.permutation(10000)
    test_data      = test_data[p][0:M]
    test_labels    = test_labels[p][0:M]

#arranging data
if (FLAGS.buffer != 'unit_conv_1_3'):
    train_data = np.concatenate((train_data_1,train_data_2,train_data_3,train_data_4,train_data_5))
train_data = train_data.reshape(train_data.shape[0], -1)
test_data  = test_data.reshape(test_data.shape[0], -1)

#rearranging and reshaping labels
train_labels = np.concatenate((train_labels_1,train_labels_2,train_labels_3,train_labels_4,train_labels_5))
train_labels = train_labels.reshape(train_labels.shape[0], -1).astype(np.float32)
test_labels  = test_labels.reshape(test_labels.shape[0], -1).astype(np.float32)

#calculating number of mini-batches
N = M // mb

#calculating k-NN
k_arr = np.array([1, 3, 5, 7, 9, 11, 13, 15, 31, 51, 101, 151, 201, 301, 401, 501])
knn = cv2.ml.KNearest_create()
print ('Start training kNN model...')
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)
print ("KNN accuracies:")
for k in np.nditer(k_arr):
    acc_avg = 0.0
    for i in range(N): #calculating N test batches
        b = i*mb
        e = (i+1)*mb
        ret, results, neighbors, dist = knn.findNearest(test_data[b:e], k)
        acc = np.mean(results == test_labels[b:e])
        acc_avg += acc
        print("k=%0d (batch %0d out of %0d): tmp accuracy = %.4f" %(k, i+1, N, acc))
    acc_avg /= N
    print("k=%0d: %.4f" %(k, acc_avg))
