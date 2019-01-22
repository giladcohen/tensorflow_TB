from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json
import scipy.optimize as opt

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(6.0, 6.0))

# setting all experiments
all_ks = [1, 3, 4, 5, 6, 7, 8, 9, 10,
          12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
          45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
          110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
          220, 240, 260, 280, 300,
          350, 400, 450, 500,
          600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
all_ks.extend(range(1600, 6001, 100))

NORM   = 'L1'
NOM = 2.426

logdir_vec  = []
n_vec       = []
max_ks      = []
num_classes = 2
for i in range(1, 11):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=08011900'.format(i))
    n_vec.append(int(i * 1000))
    max_ks.append(int(i * 1000 / num_classes))

n_vec = np.array(n_vec)
max_ks = np.array(max_ks)

if NORM == 'L1':
    C_vec = 1e-4 * np.array([9.611, 6.989, 6.308, 6.192, 5.466, 5.391, 5.871, 5.16, 5.273, 5.379])
elif NORM == 'L2':
    C_vec = np.array([0.02066, 0.01482, 0.01367, 0.0141, 0.0122, 0.01281, 0.01319, 0.01183, 0.01197, 0.01156])
else:
    raise AssertionError("No such metric {}".format(NORM))


# we need to fit C*
if NORM == 'L1':
    C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.00056, 0.000543, 0.000532, 0.000525, 0.00052])
elif NORM == 'L2':
    pass
    # C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.000565, 0.00055, 0.00053, 0.000525, 0.00052])
else:
    raise AssertionError('No such norm {}'.format(NORM))

# plotting the C_vec and its fitted version:
ax = fig.add_subplot(111)
ax.set_ylabel('C', labelpad=5, fontdict={'fontsize': 12})
ax.set_xlabel('number of samples')
ax.yaxis.grid()
ax.plot(n_vec, C_vec, 'ko')
ax.plot(n_vec, C_vec_fitted, '--r')


plt.tight_layout()
plt.savefig('fit_lipschitz_constant.png')


