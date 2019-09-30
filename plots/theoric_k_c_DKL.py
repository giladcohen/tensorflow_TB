from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from tensorflow_TB.utils.plots import load_data_from_csv_wrapper, add_subplot_axes
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

NORM   = 'L1'
NOM = 2.426
num_classes = 2

logdir_vec  = []
n_vec       = []
max_ks      = []
for i in range(1, 11):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=08011900'.format(i))
    n_vec.append(int(i * 1000))
    max_ks.append(int(i * 1000 / num_classes))

n_vec = np.array(n_vec)
max_ks = np.array(max_ks)

# calc k
measure = 'norm_{}_knn_kl_div2_avg'.format(NORM)
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
    C_vec = 1e-4 * np.array([9.611, 6.989, 6.308, 6.192, 5.466, 5.391, 5.871, 5.16, 5.273, 5.379])
elif NORM == 'L2':
    C_vec = np.array([0.02066, 0.01482, 0.01367, 0.0141, 0.0122, 0.01281, 0.01319, 0.01183, 0.01197, 0.01156])
else:
    raise AssertionError("No such metric {}".format(NORM))

# fit C
if NORM == 'L1':
    C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.00056, 0.000543, 0.000532, 0.000525, 0.00052])
elif NORM == 'L2':
    pass
    # C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.000565, 0.00055, 0.00053, 0.000525, 0.00052])
else:
    raise AssertionError('No such norm {}'.format(NORM))

optimal_k_cat_v_dogs        = optimal_k
optimal_k_fitted_cat_v_dogs = optimal_k_fitted
C_vec_cat_v_dogs            = C_vec
C_vec_fitted_cats_v_dogs    = C_vec_fitted
DKL_cats_v_dogs             = knn_score


logdir_vec  = []
n_vec       = []
max_ks      = []
for i in range(1, 11):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10_airplanes_v_ships/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=21011900'.format(i))
    n_vec.append(int(i * 1000))
    max_ks.append(int(i * 1000 / num_classes))

n_vec = np.array(n_vec)
max_ks = np.array(max_ks)

# calc k
measure = 'norm_{}_knn_kl_div2_avg'.format(NORM)
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
    C_vec = 1e-4 * np.array([10, 7.1, 6.8, 4.4, 4.9, 4.1, 4.2, 3.85, 3.8, 3.55])
elif NORM == 'L2':
    pass
    # C_vec = np.array([0.02066, 0.01482, 0.01367, 0.0141, 0.0122, 0.01281, 0.01319, 0.01183, 0.01197, 0.01156])
else:
    raise AssertionError("No such metric {}".format(NORM))

# # fit C
if NORM == 'L1':
    C_vec_fitted = np.array([C_vec[0], C_vec[1], 0.000598, 0.00052, 0.00047, 0.00043, 0.000405, C_vec[7], 0.000375, 0.00037])
elif NORM == 'L2':
    pass
    # C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.000565, 0.00055, 0.00053, 0.000525, 0.00052])
else:
    raise AssertionError('No such norm {}'.format(NORM))

optimal_k_airplanes_v_ships        = optimal_k
optimal_k_fitted_airplanes_v_ships = optimal_k_fitted
C_vec_airplanes_v_ships            = C_vec
C_vec_fitted_airplanes_v_ships     = C_vec_fitted
DKL_airplanes_v_ships              = knn_score


fig = plt.figure(figsize=(10.0, 8.0))
ax1 = fig.add_subplot(321)
ax1.plot(n_vec, DKL_cats_v_dogs, 'ko')
ax1.set_ylabel('$D_{KL}$($k$-NN||DNN)', labelpad=5, fontdict={'fontsize': 16})
ax1.yaxis.grid(True)
ax1.set_xticklabels([])
ax1.tick_params(labelsize=14)
ax1.set_title('Cats vs Dogs', fontdict={'fontsize': 14})

ax2 = fig.add_subplot(322)
ax2.plot(n_vec, DKL_airplanes_v_ships, 'ko')
ax2.set_ylabel('$D_{KL}$($k$-NN||DNN)', labelpad=5, fontdict={'fontsize': 16})
ax2.yaxis.grid(True)
ax2.set_xticklabels([])
ax2.tick_params(labelsize=14)
ax2.set_title('Airplanes vs Ships', fontdict={'fontsize': 14})

ax3 = fig.add_subplot(323)
ax3.set_ylabel('$k^*$', labelpad=5, fontdict={'fontsize': 16})
ax3.plot(n_vec, optimal_k_cat_v_dogs, 'ko')
ax3.plot(n_vec, optimal_k_fitted_cat_v_dogs, '--r')
ax3.grid(True)
ax3.set_xticklabels([])
ax3.set_yticks([0, 1000, 2000, 3000, 4000])
ax3.tick_params(labelsize=14)

ax4 = fig.add_subplot(324)
ax4.set_ylabel('$k^*$', labelpad=5, fontdict={'fontsize': 16})
ax4.plot(n_vec, optimal_k_airplanes_v_ships, 'ko')
ax4.plot(n_vec, optimal_k_fitted_airplanes_v_ships, '--r')
ax4.grid(True)
ax4.set_xticklabels([])
ax4.set_yticks([0, 1000, 2000, 3000, 4000, 5000])
ax4.tick_params(labelsize=14)

ax5 = fig.add_subplot(325)
ax5.set_ylabel('C', labelpad=5, fontdict={'fontsize': 16})
ax5.set_xlabel('number of samples', fontdict={'fontsize': 14})
ax5.grid()
ax5.plot(n_vec, C_vec_cat_v_dogs, 'ko')
ax5.plot(n_vec, C_vec_fitted_cats_v_dogs, '--r')
ax5.tick_params(labelsize=14)

ax6 = fig.add_subplot(326)
ax6.set_ylabel('C', labelpad=5, fontdict={'fontsize': 16})
ax6.set_xlabel('number of samples', fontdict={'fontsize': 14})
ax6.grid()
ax6.plot(n_vec, C_vec_airplanes_v_ships, 'ko')
ax6.plot(n_vec, C_vec_fitted_airplanes_v_ships, '--r')
ax6.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('theoretic_k_c_DKL.png', dpi=350)


