'''
Adding comparison of individual network accuracy with KNN accuracy.
This code improves test5.py.
'''

import numpy as np
import tensorflow as tf
import os.path
import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10, cifar100 #for debug

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('mode', 'majority_vote', 'Mode of estimating the true label')
flags.DEFINE_string('log_root', '', 'log dir of the net to use')
flags.DEFINE_integer('K', '-1', 'number of K nearest neighbors')
flags.DEFINE_string('network', 'gap', 'mp or gap')
flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')

#BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
#DATA_PATH = os.path.join(BASE_PATH, 'tmp')
DATA_PATH = '/data/gilad/Resnet_KNN/resnet_dump'

if (FLAGS.log_root != ''):
    logs_vec = [FLAGS.log_root]
else:
    if (FLAGS.network == 'mp'):
        logs_vec = ['logs/logs_wrn28-10_0620_290417' , \
                    'logs/logs_wrn28-10_0700_290417' , \
                    'logs/logs_wrn28-10_0701_290417' , \
                    'logs/logs_wrn28-10_0927_300417' , \
                    'logs/logs_wrn28-10_0928_300417' , \
                    'logs/logs_wrn28-10_0929_300417' , \
                    'logs/logs_wrn28-10_1308_010517' , \
                    'logs/logs_wrn28-10_1309_010517' , \
                    'logs/logs_wrn28-10_1310_010517' ]
    elif (FLAGS.network == 'gap'):
        if (FLAGS.dataset == 'cifar10'):
            logs_vec = ['logs/logs_wrn28-10_1513_220317' , \
                        'logs/logs_wrn28-10_0137_050417' , \
                        'logs/logs_wrn28-10_0203_090417' , \
                        'logs/logs_wrn28-10_0853_120417' , \
                        'logs/logs_wrn28-10_0732_140417' , \
                        'logs/logs_wrn28-10_1434_160417' , \
                        'logs/logs_wrn28-10_1753_170417' ]
        elif (FLAGS.dataset == 'cifar100'):
            logs_vec = ['logs/logs_wrn28-10_2101_110517' , \
                        'logs/logs_wrn28-10_2102_110517' , \
                        'logs/logs_wrn28-10_2103_110517' , \
                        'logs/logs_wrn28-10_1944_120517' , \
                        'logs/logs_wrn28-10_1945_120517' , \
                        'logs/logs_wrn28-10_1946_120517' , \
                        'logs/logs_wrn28-10_1546_140517' , \
                        'logs/logs_wrn28-10_1547_140517' , \
                        'logs/logs_wrn28-10_1548_140517' ]

N_logs = len(logs_vec)
if  (FLAGS.dataset == 'cifar10'):
    NUM_CLASSES = 10
elif(FLAGS.dataset == 'cifar100'):
    NUM_CLASSES = 100

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

#assertions
# for i in range(1,N_logs):
#     print('asserting i=%0d for images/labels' %i)
#     assert (train_images[i]   == train_images[0]).all()
#     assert (train_labels[i]   == train_labels[0]).all()
#     assert (test_images[i]    == test_images[0]).all()
#     assert (test_labels[i]    == test_labels[0]).all()
#  
# for i in range(N_logs):
#     print('asserting i=%0d for recorded data' %i)
#     assert np.sum(train_logits[i]   == -1) == 0
#     assert np.sum(train_fc1[i]      == -1) == 0
#     assert np.sum(test_logits[i]    == -1) == 0
#     assert np.sum(test_fc1[i]       == -1) == 0
    
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

if (FLAGS.K != -1):
    k_arr = np.array([FLAGS.K])
else:
    k_arr = np.array([1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000,3000,4000,5000])
train_data = train_fc1
test_data  = test_fc1

def err_report(label_est, true_labels, label_vote=None):
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


if (FLAGS.mode == 'majority_vote'):
    label_mat  = np.hstack(test_predictions_reshaped) #yields (10000, 9) matrix. Every row has 9 different estimations
    label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
    label_est  = np.argmax(label_vote, axis=1)
    tie_indices, err_indices = err_report(label_est, test_labels[0], label_vote)
    accuracy = np.average(label_est == test_labels[0])
    print('Accuracy for majority vote method: %.4f' %accuracy)
else: #we need to train KNN
    kNN_vec = []
    for i in range(N_logs): #train
        knn = cv2.ml.KNearest_create()
        print ('Start training kNN model for i=%0d for %0s' %(i, FLAGS.mode))
        knn.train(train_data[i], cv2.ml.ROW_SAMPLE, train_labels_reshaped[i])
        kNN_vec.append(knn)
    for k in np.nditer(k_arr):
        neighbors_vec=[]
        dist_vec=[] # for 'fc1_dist"
        consistency_level_vec=[] # for fc1_cons
        common_label_vec=[] # for fc1_cons
        democ_pred_mat=np.empty([N_test, N_logs], dtype=np.int) #for 'democracy'
        first_choice=np.empty([N_test, N_logs], dtype=np.int) #for 'global democracy'
        second_choice=np.empty([N_test, N_logs], dtype=np.int) #for 'global democracy'
        for i in range(N_logs): #evaluation
            print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
            ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
            neighbors = neighbors.astype(np.int)
            neighbors_vec.append(neighbors)
            dist_vec.append(dist)
            if (FLAGS.mode == 'fc1_cons'):
                hist = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=neighbors)
                consistency_level = np.max(hist, axis=1).reshape([N_test, 1])
                common_label = np.argmax(hist, axis=1).reshape([N_test, 1])
                consistency_level_vec.append(consistency_level)
                common_label_vec.append(common_label)
            if (FLAGS.mode == 'democracy'):
                hist          = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=neighbors)
                top_neighbors = np.apply_along_axis(lambda x: np.argsort(x)[::-1], axis=1, arr=hist)
                top_logits    = np.apply_along_axis(lambda x: np.argsort(x)[::-1], axis=1, arr=test_logits[i])
                for row in range(N_test):
                    if (top_neighbors[row,0] == top_logits[row,0] or
                        top_neighbors[row,0] == top_logits[row,1]):
                        democ_pred_mat[row,i] = top_neighbors[row,0]
                    else:
                        democ_pred_mat[row,i] = test_predictions_reshaped[i][row]
            if (FLAGS.mode == 'global_democracy'):
                first_choice[:,i]  = np.apply_along_axis(lambda x: np.argsort(x)[::-1][0], axis=1, arr=test_logits[i])
                second_choice[:,i] = np.apply_along_axis(lambda x: np.argsort(x)[::-1][1], axis=1, arr=test_logits[i])
        if (FLAGS.mode == 'fc1_vote'):
            label_mat  = np.hstack(neighbors_vec)
            label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
            label_est  = np.argmax(label_vote, axis=1)
        elif (FLAGS.mode == 'fc1_dist'):
            neighbors_all = np.hstack(neighbors_vec) #like label_mat
            dist_mat      = np.hstack(dist_vec).astype(np.float32)
            dist_min_ind  = np.argpartition(dist_mat, k, axis=1)[:,0:k]
            label_mat = np.empty([N_test, k])
            for row in range(N_test):
                label_mat[row] = neighbors_all[row][dist_min_ind[row]]
            label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
            label_est  = np.argmax(label_vote, axis=1)
        elif (FLAGS.mode == 'fc1_cons'):
            consistency_level_mat = np.hstack(consistency_level_vec)
            common_label_mat = np.hstack(common_label_vec)
            label_est = np.empty([N_test])
            for row in range(N_test):
                ind_max = np.where(consistency_level_mat[row] == consistency_level_mat[row].max())[0]
                labels  = common_label_mat[row][ind_max]
                label_est[row] = np.bincount(labels).argmax()
            label_vote = None
        elif (FLAGS.mode == 'democracy'):
            label_mat  = democ_pred_mat #yields (10000, 9) matrix. Every row has 9 different estimations
            label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
            label_est  = np.argmax(label_vote, axis=1)
            accuracy = np.average(label_est == test_labels[0])
        elif (FLAGS.mode == 'global_democracy'):
            neighbors_all  = np.hstack(neighbors_vec)
            hist = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=neighbors_all)
            hist_first_choices = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=first_choice)
            consistency_level = np.max(hist_first_choices, axis=1).reshape([N_test, 1])
            label_mat = np.empty([N_test, N_logs], dtype=np.int)
            label_est = np.empty([N_test], dtype=np.int)
            for row in range(N_test):
                for i in range(N_logs):
                    if (consistency_level[row]>6):
                        label_mat[row,i] = first_choice[row][i]
                    else:
                        if (hist[row][second_choice[row,i]] > 1.5*hist[row][first_choice[row,i]]):
                            label_mat[row,i] = second_choice[row][i]
                        else:
                            label_mat[row,i] = first_choice[row][i]
            label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
            label_est  = np.argmax(label_vote, axis=1)
            accuracy = np.average(label_est == test_labels[0])
        else:
            raise ValueError("mode %0s is not expected." %FLAGS.mode) 
        tie_indices, err_indices = err_report(label_est, test_labels[0], label_vote)
        accuracy = np.average(label_est == test_labels[0])
        print('Accuracy for mode %0s for k=%0d: %.4f' %(FLAGS.mode, k, accuracy))
        assert (round(1 - accuracy, 4)) == float(len(err_indices))/N_test
