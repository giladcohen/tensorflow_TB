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

# setting all experiments
all_ks = [1, 3, 4, 5, 6, 7, 8, 9, 10,
          12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
          45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
          110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
          220, 240, 260, 280, 300,
          350, 400, 450, 500,
          600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
all_ks.extend(range(1600, 6001, 100))

NORM   = 'L2'
NOM = 2.426

logdir_vec  = []
n_vec       = []
max_ks      = []
num_classes = 2
for i in range(1, 11):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=21011900'.format(i))
    n_vec.append(int(i * 1000))
    max_ks.append(int(i * 1000 / num_classes))

n_vec = np.array(n_vec)
max_ks = np.array(max_ks)

# calc k
measure = 'norm_{}_knn_kl_div4_median'.format(NORM)
knn_score = []
optimal_k = []
for i, root_dir in enumerate(logdir_vec):
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    max_k = max_ks[i]
    best_score = np.inf  # lower is better
    best_k = None
    with open(json_file) as f:
        data = json.load(f)
    for k in all_ks:
        if k <= max_k:
            m_str = 'knn_k_{}_{}'.format(k, measure)
            score = data['test']['regular'][m_str]['values'][0]
            if score < best_score:
                best_score = score
                best_k = k
    knn_score.append(best_score)
    optimal_k.append(best_k)

# fitting k
A = np.vstack([n_vec, np.ones(len(n_vec))]).T
m, c = np.linalg.lstsq(A, optimal_k, rcond=None)[0]
optimal_k_fitted = n_vec * m + c

# calc C
if NORM == 'L1':
    C_vec = 1e-4 * np.array([12, 9.2, 7, 5.9, 5.4, 4.2, 3.8, 3.75, 3.4, 3.3])
elif NORM == 'L2':
    pass
    # C_vec = np.array([0.02066, 0.01482, 0.01367, 0.0141, 0.0122, 0.01281, 0.01319, 0.01183, 0.01197, 0.01156])
else:
    raise AssertionError("No such metric {}".format(NORM))

# # fit C
# if NORM == 'L1':
#     C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.00056, 0.000543, 0.000532, 0.000525, 0.00052])
# elif NORM == 'L2':
#     pass
#     # C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.000565, 0.00055, 0.00053, 0.000525, 0.00052])
# else:
#     raise AssertionError('No such norm {}'.format(NORM))

# plotting the C_vec and its fitted version:
fig = plt.figure(figsize=(6.0, 6.0))
ax1 = fig.add_subplot(211)
ax1.set_ylabel('$k^*$', labelpad=5, fontdict={'fontsize': 12})
ax1.set_xlabel('number of samples')
ax1.plot(n_vec, optimal_k, 'ko')
ax1.plot(n_vec, optimal_k_fitted, '--r')
ax1.yaxis.grid()

ax1.get_xaxis().set_visible(False)
plt.show()

# ax2 = fig.add_subplot(212)
# ax2.set_ylabel('C', labelpad=5, fontdict={'fontsize': 12})
# ax2.set_xlabel('number of samples')
# ax2.yaxis.grid()
# ax2.plot(n_vec, C_vec, 'ko')

# ax2.plot(n_vec, C_vec_fitted, '--r')
#
# plt.tight_layout()
# plt.savefig('opt_k_and_c.png')


