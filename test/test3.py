'''
Using ensemble of 7 WRN-28-10 for better accuracy. 
'''

import numpy as np
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10 #for debug

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('mode', 'pool_out_cons', 'Mode of estimating the true label')
flags.DEFINE_boolean('reuse_typ_dist', True, 'reuse pre-calculated typical dist')
flags.DEFINE_integer('nets', 7, 'number of nets to use')

BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
DATA_PATH = os.path.join(BASE_PATH, 'tmp')

logs_vec = ['logs/logs_wrn28-10_1513_220317' , \
            'logs/logs_wrn28-10_0137_050417' , \
            'logs/logs_wrn28-10_0203_090417' , \
            'logs/logs_wrn28-10_0853_120417' , \
            'logs/logs_wrn28-10_0732_140417' , \
            'logs/logs_wrn28-10_1434_160417' , \
            'logs/logs_wrn28-10_1753_170417' ]
N_logs = len(logs_vec)
N_logs_c = min(N_logs , FLAGS.nets)

train_images=[]
train_labels=[]
train_logits=[]
train_pool_in=[]
train_pool_out=[]
train_labels_reshaped=[]
train_logits_reshaped=[]

test_images=[]
test_labels=[]
test_logits=[]
test_pool_in=[]
test_pool_out=[]
test_labels_reshaped=[]
test_logits_reshaped=[]

for i in range(N_logs):
    print('reading data for i=%0d' %i)
    suffix = logs_vec[i][-12:]
    train_images.append(np.load(os.path.join(DATA_PATH, 'train_images_raw'+suffix+'.npy')))
    train_labels.append(np.load(os.path.join(DATA_PATH, 'train_labels'+suffix+'.npy')))
    train_logits.append(np.load(os.path.join(DATA_PATH, 'train_logits'+suffix+'.npy')))
    train_pool_in.append(np.load(os.path.join(DATA_PATH, 'train_pool_in'+suffix+'.npy'), mmap_mode='r'))
    train_pool_out.append(np.load(os.path.join(DATA_PATH, 'train_pool_out'+suffix+'.npy')))
    test_images.append(np.load(os.path.join(DATA_PATH, 'test_images_raw'+suffix+'.npy')))
    test_labels.append(np.load(os.path.join(DATA_PATH, 'test_labels'+suffix+'.npy')))
    test_logits.append(np.load(os.path.join(DATA_PATH, 'test_logits'+suffix+'.npy')))
    test_pool_in.append(np.load(os.path.join(DATA_PATH, 'test_pool_in'+suffix+'.npy'), mmap_mode='r'))
    test_pool_out.append(np.load(os.path.join(DATA_PATH, 'test_pool_out'+suffix+'.npy')))

# assertions
# for i in range(1,N_logs):
#     print('asserting i=%0d for images/labels' %i)
#     assert (train_images[i]   == train_images[0]).all()
#     assert (train_labels[i]   == train_labels[0]).all()
#     assert (test_images[i]    == test_images[0]).all()
#     assert (test_labels[i]    == test_labels[0]).all()

# for i in range(N_logs):
#     print('asserting i=%0d for recorded data' %i)
#     assert np.sum(train_logits[i]   == -1) == 0
#     assert np.sum(train_pool_in[i]  == -1) < 100
#     assert np.sum(train_pool_out[i] == -1) == 0
#     assert np.sum(test_logits[i]    == -1) == 0
#     assert np.sum(test_pool_in[i]   == -1) < 20
#     assert np.sum(test_pool_out[i]  == -1) == 0
    
#debug
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

#reshaping logits and labels to (10000,1) arrays instead of (10000,) arrays
N_train = train_labels[0].shape[0]
N_test  = test_labels[0].shape[0]
for i in range(N_logs):
    train_labels_reshaped.append(train_labels[i].reshape(N_train, 1).astype(np.float32))
    train_logits_reshaped.append(train_logits[i].reshape(N_train, 1).astype(np.float32))
    test_labels_reshaped.append(test_labels[i].reshape(N_test,    1).astype(np.float32))
    test_logits_reshaped.append(test_logits[i].reshape(N_test,    1).astype(np.float32))

tie_indices=[]
err_indices=[]
k_arr = np.array([1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000,3000,4000,5000])

if (FLAGS.mode[0:-5] == 'pool_out'):
    train_data = train_pool_out
    test_data  = test_pool_out
elif (FLAGS.mode[0:-5] == 'pool_in'):
    train_data = []
    test_data  = []
    for i in range(N_logs_c):
        train_data.append(train_pool_in[i].reshape([N_train, -1]))
        test_data.append(test_pool_in[i].reshape([N_test, -1]))

if (FLAGS.mode == 'majority_vote'):
    label_mat  = np.hstack(test_logits_reshaped).astype(np.int) #yields (10000, 7) matrix. Every row has 7 different estimations
    label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=10), axis=1, arr=label_mat)
    label_est  = np.argmax(label_vote, axis=1)
    # warn me for samples that yield a tie between 2 (or more) logits
    for ind in range(N_test):
        if (len(np.nonzero(label_vote[ind]==max(label_vote[ind]))[0])>1):
            print("There is a tie for test sample #%0d" %ind)
            tie_indices.append(ind)
        if (label_est[ind] != test_labels[0][ind]):
            print("There is an error for test sample #%0d" %ind)
            err_indices.append(ind)
    accuracy = np.average(label_est == test_labels[0])
    print('Accuracy for majority vote method: %.4f' %accuracy)
    
elif ((FLAGS.mode == 'pool_out_vote') or (FLAGS.mode == 'pool_in_vote')):
    kNN_vec = []
    for i in range(N_logs_c): #train
        knn = cv2.ml.KNearest_create()
        print ('Start training kNN model for i=%0d for %0s' %(i, FLAGS.mode))
        knn.train(train_data[i], cv2.ml.ROW_SAMPLE, train_labels_reshaped[i])
        kNN_vec.append(knn)
    for k in np.nditer(k_arr):
        tie_indices=[] #TODO(implement list for all k's)
        err_indices=[]
        neighbors_vec = []
        for i in range(N_logs_c): #evaluation
            print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
            ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
            neighbors_vec.append(neighbors)
        label_mat  = np.hstack(neighbors_vec).astype(np.int)
        label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=10), axis=1, arr=label_mat)
        label_est  = np.argmax(label_vote, axis=1)
        # warn me for samples that yield a tie between 2 (or more) logits
        for ind in range(N_test):
            if (len(np.nonzero(label_vote[ind]==max(label_vote[ind]))[0])>1):
                #print("There is a tie for test sample #%0d" %ind)
                tie_indices.append(ind)
            if (label_est[ind] != test_labels[0][ind]):
                #print("There is an error for test sample #%0d" %ind)
                err_indices.append(ind)
        accuracy = np.average(label_est == test_labels[0])
        print('Accuracy for mode %0s for k=%0d: %.4f' %(FLAGS.mode, k, accuracy))
        assert (round(1 - accuracy, 4)) == float(len(err_indices))/N_test
        
elif ((FLAGS.mode == 'pool_out_dist') or (FLAGS.mode == 'pool_in_dist')):
    kNN_vec = []
    for i in range(N_logs_c): #train
        knn = cv2.ml.KNearest_create()
        print ('Start training kNN model for i=%0d for %0s' %(i, FLAGS.mode))
        knn.train(train_data[i], cv2.ml.ROW_SAMPLE, train_labels_reshaped[i])
        kNN_vec.append(knn)
    if (FLAGS.reuse_typ_dist):
        print ('setting pre-calculated typical dists for nets.')
        typical_dist = [163.76894, 267.77451, 253.31615, 198.35704, 209.6272, 176.71906, 10.141984]
    else:
        typical_dist = []
        for i in range(N_logs_c): #evaluation
            print ('Start evaluating kNN model for i=%0d, k=1 in %0s - for typical_dist' %(i,FLAGS.mode))
            ret, results, neighbors, dist = knn.findNearest(test_data[i], 1)
            typical_dist.append(np.average(dist))
    for k in np.nditer(k_arr):
        tie_indices=[] #TODO(implement list for all k's)
        err_indices=[]
        neighbors_vec = []
        dist_vec = []
        for i in range(N_logs_c): #evaluation
            print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
            ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
            dist_norm = dist/typical_dist[i]
            neighbors_vec.append(neighbors)
            dist_vec.append(dist_norm)
        neighbors_all = np.hstack(neighbors_vec).astype(np.int)
        dist_mat      = np.hstack(dist_vec).astype(np.float32)
        dist_min_ind  = np.argpartition(dist_mat, k, axis=1)[:,0:k]
        label_mat = np.empty([N_test, k])
        for row in range(N_test):
            label_mat[row] = neighbors_all[row][dist_min_ind[row]]
        label_mat = label_mat.astype(np.int)
        label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=10), axis=1, arr=label_mat)
        label_est  = np.argmax(label_vote, axis=1)
        # warn me for samples that yield a tie between 2 (or more) logits
        for ind in range(N_test):
            if (len(np.nonzero(label_vote[ind]==max(label_vote[ind]))[0])>1):
                #print("There is a tie for test sample #%0d" %ind)
                tie_indices.append(ind)
            if (label_est[ind] != test_labels[0][ind]):
                #print("There is an error for test sample #%0d" %ind)
                err_indices.append(ind)
        accuracy = np.average(label_est == test_labels[0])
        print('Accuracy for mode %0s for k=%0d: %.4f' %(FLAGS.mode, k, accuracy))
        assert (round(1 - accuracy, 4)) == float(len(err_indices))/N_test

elif ((FLAGS.mode == 'pool_out_cons') or (FLAGS.mode == 'pool_in_cons')):
    kNN_vec = []
    for i in range(N_logs_c): #train
        knn = cv2.ml.KNearest_create()
        print ('Start training kNN model for i=%0d for %0s' %(i, FLAGS.mode))
        knn.train(train_data[i], cv2.ml.ROW_SAMPLE, train_labels_reshaped[i])
        kNN_vec.append(knn)
    for k in np.nditer(k_arr):
        #tie_indices=[]
        err_indices=[]
        consistency_level_vec = []
        common_label_vec = []
        for i in range(N_logs_c): #evaluation
            print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
            ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
            neighbors = neighbors.astype(np.int)
            hist = np.apply_along_axis(lambda x: np.bincount(x, minlength=10), axis=1, arr=neighbors)
            consistency_level = np.max(hist, axis=1).reshape([N_test, 1])
            common_label = np.argmax(hist, axis=1).reshape([N_test, 1])
            consistency_level_vec.append(consistency_level)
            common_label_vec.append(common_label)
        consistency_level_mat = np.hstack(consistency_level_vec)
        common_label_mat = np.hstack(common_label_vec)
        label_est = np.empty([N_test])
        for i in range(N_test):
            ind_max = np.where(consistency_level_mat[i] == consistency_level_mat[i].max())[0]
            labels  = common_label_mat[i][ind_max]
            label_est[i] = np.bincount(labels).argmax()
            if (label_est[i] != test_labels[0][i]):
                #print("There is an error for test sample #%0d" %i)
                err_indices.append(i)
        accuracy = np.average(label_est == test_labels[0])
        print('Accuracy for mode %0s for k=%0d: %.4f' %(FLAGS.mode, k, accuracy))
        assert (round(1 - accuracy, 4)) == float(len(err_indices))/N_test
