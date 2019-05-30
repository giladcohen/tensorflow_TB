from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os
import scipy.io

SVHN_PATH      = '/data/dataset/SVHN/format2'
SVHN_MINI_PATH = '/data/dataset/SVHN_MINI'

superseed = 15101985
rand_gen = np.random.RandomState(superseed)

# trainset
data_dict = scipy.io.loadmat(os.path.join(SVHN_PATH, 'train_32x32.mat'))
data, label = data_dict['X'], data_dict['y']
data = np.moveaxis(data, -1, 0)
# data = data.astype(np.float32)
np.place(label, label == 10, 0)

new_train_inds = rand_gen.choice(np.arange(len(label)), 50000, replace=False)
new_train_inds.sort()
X_train = data[new_train_inds]
y_train = label[new_train_inds]

# test set
data_dict = scipy.io.loadmat(os.path.join(SVHN_PATH, 'test_32x32.mat'))
data, label = data_dict['X'], data_dict['y']
data = np.moveaxis(data, -1, 0)
# data = data.astype(np.float32)
np.place(label, label == 10, 0)

new_test_inds = rand_gen.choice(np.arange(len(label)), 10000, replace=False)
new_test_inds.sort()
X_test = data[new_test_inds]
y_test = label[new_test_inds]

np.save(os.path.join(SVHN_MINI_PATH, 'X_train.npy'), X_train)
np.save(os.path.join(SVHN_MINI_PATH, 'X_test.npy') , X_test)
np.save(os.path.join(SVHN_MINI_PATH, 'y_train.npy'), y_train)
np.save(os.path.join(SVHN_MINI_PATH, 'y_test.npy') , y_test)



# DEBUG
# for i in range(10):
#     print('i={}: {}   {}'.format(i, np.sum(y_train[train_inds] == i)/len(train_inds), np.sum(y_train[val_inds] == i)/len(val_inds)))
#     print('i={}: {}   {}'.format(i, np.sum(y_val_sparse == i)/len(y_val_sparse), np.sum(y_train_sparse == i)/len(y_train_sparse)))
#     print('i={}: {}   {}'.format(i, np.sum(y_train == i)/len(y_train), np.sum(y_train[val_indices] == i)/len(val_indices)))

