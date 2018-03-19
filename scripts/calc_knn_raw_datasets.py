from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import numpy as np

def get_data(dataset_name):
    if dataset_name == 'cifar100':
        data = tf.keras.datasets.cifar100
    elif dataset_name == 'cifar10':
        data = tf.keras.datasets.cifar10
    elif dataset_name == 'mnist':
        data = tf.keras.datasets.mnist
    else:
        err_str = 'dataset {} is not legal'.format(dataset_name)
        raise AssertionError(err_str)

    (X_train, y_train), (X_test, y_test) = data.load_data()

    if dataset_name in ['cifar10', 'cifar100']:
        y_train = np.squeeze(y_train, axis=1)
        y_test = np.squeeze(y_test, axis=1)
    if dataset_name == 'mnist':
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

    return (X_train, y_train), (X_test, y_test)

knn_vec = [KNeighborsClassifier(
              n_neighbors=k,
              weights='uniform',
              p=1,
              n_jobs=20) for k in [1, 5, 30, 100]]

datasets = ['cifar10', 'cifar100', 'mnist']

for dataset in datasets:
    for knn in knn_vec:
        print('Calculating KNN score for dataset {} with K={}'.format(dataset, knn.n_neighbors))
        (X_train, y_train), (X_test, y_test) = get_data(dataset)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        knn.fit(X_train, y_train)
        y_pred_prob = knn.predict_proba(X_test)
        y_pred = y_pred_prob.argmax(axis=1)

        accuracy = np.sum(y_pred == y_test) / X_test.shape[0]
        print('\t\taccuracy = {}'.format(accuracy))
print('done')
