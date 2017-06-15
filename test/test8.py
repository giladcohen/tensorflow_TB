'''
Adding advanced variations for fc1_dist method: raw, norm1, norm2
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
flags.DEFINE_boolean('reuse_typ_dist', True, 'reuse pre-calculated typical dist')
flags.DEFINE_string('dist', 'typ', 'method for calculating normalized k-NN distance. Can be raw/typ/norm1/norm2')

print('Runnint KNN analysis script for: mode=%s, log_root=%s, K=%0d, network=%s, dataset=%s, reuse_typ_dist=%r, dist=%s' \
            %(FLAGS.mode, FLAGS.log_root, FLAGS.K, FLAGS.network, FLAGS.dataset, FLAGS.reuse_typ_dist, FLAGS.dist))

#BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
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
            typical_dist_precalculated = [10.6885, 8.8676, 9.0507, \
                                           9.8238, 9.5307, 10.766503, 10.1419]
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
            typical_dist_precalculated = [205.8570, 204.9845, 204.5049, \
                                          205.1015, 205.6346, 206.8413, \
                                          208.3040, 199.5526, 200.4931]

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
train_predictions_reshaped=[]

test_images=[]
test_labels=[]
test_logits=[]
test_predictions=[]
test_fc1=[]
test_predictions_reshaped=[]

BBP_vec = []
kNN_vec = []

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
    #BBP_vec.append(np.load(os.path.join(DATA_PATH,      'BBP'+FLAGS.log_root[-12:]+'.npy')))

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
if   (FLAGS.dataset == 'cifar10'):
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

#BBP = np.zeros([N_test, N_train], dtype=np.int)
for i in range(N_logs):
    train_predictions_reshaped.append(train_predictions[i].reshape(N_train, 1).astype(np.int))
    test_predictions_reshaped.append(test_predictions[i].reshape(N_test,    1).astype(np.int))
    #BBP += BBP_vec[i]

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

def build_KNN(data, labels):
    ''' 
    wrapper for conviniently building all KNN models for every
    network in the ensemble.
    '''
    kNN_vec = []
    for i in range(N_logs): #train
        knn = cv2.ml.KNearest_create()
        print ('Start training kNN model for i=%0d for %0s' %(i, FLAGS.mode))
        knn.train(data[i], cv2.ml.ROW_SAMPLE, labels)
        kNN_vec.append(knn)
    return kNN_vec

def print_acc(label_est, test_labels, label_vote=None, k=0):
    '''Printing the accuracy obtained for using k'''
    accuracy = np.average(label_est == test_labels)
    tie_indices, err_indices = err_report(label_est, test_labels, label_vote)
    print('Accuracy for mode %0s for k=%0d: %.4f' %(FLAGS.mode, k, round(accuracy,4)))
    assert (round(1 - accuracy, 4)) == float(len(err_indices))/N_test
    return accuracy, tie_indices, err_indices

def norm_dist(dist, data, net_idx, method):
    dist_norm = np.empty(dist.shape)
    if (method == 'raw'):
        dist_norm = dist
    elif (method == 'typ'):
        dist_norm = dist/typical_dist[net_idx]
    elif (method == 'norm1'):
        #dist_norm = np.apply_along_axis(lambda x: x/x[0] , axis=1, arr=dist)
        for row in xrange(N_test):
            if (dist[row,0] == 0):
                print ("dist[row,0] = 0 for row=%0d, in net_idx=%0d. dividing by dist[row,1] for k>1" %(row, net_idx))
                if (dist.shape[1] > 1):
                    dist_norm[row] = dist[row]/dist[row,1]
                else: #(k=1)
                    dist_norm[row] = dist[row,0]
            else:
                dist_norm[row] = dist[row]/dist[row,0]
    elif (method == 'norm2'):
        for row in xrange(N_test):
            dist_norm[row] = dist[row]/np.linalg.norm(test_data[net_idx][row])
    return dist_norm

#many small definitions for the KNN calculations:
neighbors_vec = [] # for fc1_vote, fc1_dist, global_democracy
dist_vec=[] # for 'fc1_dist"
typical_dist = [] #for fc1_dist
consistency_level_vec=[] # for fc1_cons
common_label_vec=[] # for fc1_cons
first_choice=np.empty([N_test, N_logs], dtype=np.int) #for 'global democracy'
second_choice=np.empty([N_test, N_logs], dtype=np.int) #for 'global democracy'
label_est = np.empty([N_test]) # for all modes
label_vote = None

# type of modes that need k-NN calculation
kNN_modes = ['fc1_cons', 'democracy', 'global_democracy', 'fc1_vote', 'fc1_dist']
#building kNN models (fast!)
kNN_vec = build_KNN(train_data, train_labels_reshaped)

if (FLAGS.mode == 'majority_vote'):
    label_mat  = np.hstack(test_predictions_reshaped) #yields (10000, N_logs) matrix. Every row has N_logs different estimations
    label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
    label_est  = np.argmax(label_vote, axis=1)
    acc, tie, err = print_acc(label_est, test_labels, label_vote)
    assert (round(1 - accuracy, 4)) == float(len(err_indices))/N_test
elif any(FLAGS.mode == s for s in kNN_modes):
    for k in np.nditer(k_arr):
        if (FLAGS.mode == 'fc1_cons'):
            for i in range(N_logs):
                print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
                knn = kNN_vec[i]
                ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
                neighbors = neighbors.astype(np.int)
                hist = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=neighbors)
                consistency_level = np.max(hist, axis=1).reshape([N_test, 1])
                common_label = np.argmax(hist, axis=1).reshape([N_test, 1])
                consistency_level_vec.append(consistency_level)
                common_label_vec.append(common_label)
            consistency_level_mat = np.hstack(consistency_level_vec)
            common_label_mat = np.hstack(common_label_vec)
            for row in range(N_test):
                ind_max = np.where(consistency_level_mat[row] == consistency_level_mat[row].max())[0]
                labels  = common_label_mat[row][ind_max]
                label_est[row] = np.bincount(labels).argmax()
        elif (FLAGS.mode == 'democracy'):
            label_mat = np.empty([N_test, N_logs], dtype=np.int)
            for i in range(N_logs):
                print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
                knn = kNN_vec[i]
                ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
                neighbors = neighbors.astype(np.int)
                hist = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=neighbors)
                top_neighbors = np.apply_along_axis(lambda x: np.argsort(x)[::-1], axis=1, arr=hist)
                top_logits    = np.apply_along_axis(lambda x: np.argsort(x)[::-1], axis=1, arr=test_logits[i])
                for row in range(N_test):
                    if (top_neighbors[row,0] == top_logits[row,0] or
                        top_neighbors[row,0] == top_logits[row,1]):
                        label_mat[row,i] = top_neighbors[row,0]
                    else:
                        label_mat[row,i] = test_predictions_reshaped[i][row]
            label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
            label_est  = np.argmax(label_vote, axis=1)
        elif (FLAGS.mode == 'global_democracy'):
            for i in range(N_logs):
                print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
                knn = kNN_vec[i]
                ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
                neighbors = neighbors.astype(np.int)
                neighbors_vec.append(neighbors)
                first_choice[:,i]  = np.apply_along_axis(lambda x: np.argsort(x)[::-1][0], axis=1, arr=test_logits[i])
                second_choice[:,i] = np.apply_along_axis(lambda x: np.argsort(x)[::-1][1], axis=1, arr=test_logits[i])
            neighbors_all  = np.hstack(neighbors_vec)
            hist = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=neighbors_all)
            hist_first_choices = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=first_choice)
            consistency_level = np.max(hist_first_choices, axis=1).reshape([N_test, 1])
            label_mat = np.empty([N_test, N_logs], dtype=np.int)
            for row in range(N_test):
                for i in range(N_logs):
                    if (consistency_level[row]>((2.0/3)*N_logs)):
                        label_mat[row,i] = first_choice[row,i] #don't touch
                    else:
                        if (hist[row,second_choice[row,i]] > 1.5*hist[row][first_choice[row,i]]):
                            label_mat[row,i] = second_choice[row][i]
                        else:
                            label_mat[row,i] = first_choice[row][i]
            label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
            label_est  = np.argmax(label_vote, axis=1)
        elif (FLAGS.mode == 'fc1_vote'):
            for i in range(N_logs): 
                print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
                knn = kNN_vec[i]
                ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
                neighbors = neighbors.astype(np.int)
                neighbors_vec.append(neighbors)
            label_mat  = np.hstack(neighbors_vec)
            label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
            label_est  = np.argmax(label_vote, axis=1)
        elif (FLAGS.mode == 'fc1_dist'):
            if ((k == 1) and (FLAGS.dist == 'typ')): #generate typical_dist
                if (FLAGS.reuse_typ_dist):
                    typical_dist = typical_dist_precalculated
                    print("fc1_dist is normalized with: typical_dist=%s" %typical_dist)
                else:
                    for i in range(N_logs):
                        print ('Start evaluating kNN model for i=%0d, k=1 in %0s - for typical_dist' %(i,FLAGS.mode))
                        knn = kNN_vec[i]
                        ret, results, neighbors, dist = knn.findNearest(test_data[i], 1)
                        nn_dist = np.average(dist)
                        print ('typical dist (nn_dist) for i=%0d is %.4f' %(i, nn_dist))
                        typical_dist.append(nn_dist)
            for i in range(N_logs):
                print ('Start evaluating kNN model for i=%0d, k=%0d in %0s' %(i,k,FLAGS.mode))
                knn = kNN_vec[i]
                ret, results, neighbors, dist = knn.findNearest(test_data[i], k)
                dist_norm = norm_dist(dist, test_data, i, FLAGS.dist)
                neighbors = neighbors.astype(np.int)
                neighbors_vec.append(neighbors)
                dist_vec.append(dist_norm)
            neighbors_all = np.hstack(neighbors_vec) #like label_mat
            dist_mat      = np.hstack(dist_vec).astype(np.float32)
            dist_min_ind  = np.argpartition(dist_mat, k, axis=1)[:,0:k]
            label_mat = np.empty([N_test, k], dtype=np.int)
            for row in range(N_test):
                label_mat[row] = neighbors_all[row][dist_min_ind[row]]
            label_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=NUM_CLASSES), axis=1, arr=label_mat)
            label_est  = np.argmax(label_vote, axis=1)
        acc, tie, err = print_acc(label_est, test_labels, label_vote, k)
else:
    raise ValueError("mode %0s is not expected." %FLAGS.mode)