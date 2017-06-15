'''Loading train and test 640 final feature maps and implement k-NN'''
import numpy as np
import os.path
import matplotlib.pyplot as plt
import cv2

BASE_PATH = '/home/gilad/workspace/Resnet_KNN/tmp'

#loading train and test data
train_images = np.load(os.path.join(BASE_PATH, 'train_images.npy'))
train_labels = np.load(os.path.join(BASE_PATH, 'train_labels.npy'))
train_pool   = np.load(os.path.join(BASE_PATH, 'train_pool.npy'))
test_images  = np.load(os.path.join(BASE_PATH, 'test_images.npy'))
test_labels  = np.load(os.path.join(BASE_PATH, 'test_labels.npy'))
test_pool    = np.load(os.path.join(BASE_PATH, 'test_pool.npy'))

#rearranging labels
train_labels = train_labels.reshape(train_labels.shape[0], -1)
test_labels  = test_labels.reshape(test_labels.shape[0], -1)


#calculating k-NN
k_arr = np.array([1, 3, 5, 7, 9, 11, 31, 51, 101, 151, 201, 301, 401, 501])
knn = cv2.ml.KNearest_create()
print ('Start training kNN model...')
knn.train(train_pool, cv2.ml.ROW_SAMPLE, train_labels)
print ("KNN accuracies:")
for k in np.nditer(k_arr):
    ret, results, neighbors, dist = knn.findNearest(test_pool, k)
    acc = np.mean(results == test_labels)
    print("k=%0d: %.4f" %(k, acc))

