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
subpos = np.array([0.35, 0.25, 0.5, 0.4])
fig = plt.figure(figsize=(15.0, 8.0))

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
C_FIT  = True
C1_FIT = 'n_1'  # n_1 or n_all
K_FIT = True
NOM = 2.426

approx_bayes_error_rate = 0.089

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

optimal_k = np.array(optimal_k)
if K_FIT:
    print("fitting k_opt to a linear curve")
    plt.figure()
    plt.plot(n_vec, optimal_k, 'b')
    plt.title("optimal k")
    plt.ylabel("k*")
    plt.xlabel("Num of training samples")
    A = np.vstack([n_vec, np.ones(len(n_vec))]).T
    m, c = np.linalg.lstsq(A, optimal_k, rcond=None)[0]
    optimal_k_fitted = n_vec * m + c
    optimal_k_fitted = (np.round(optimal_k_fitted)).astype(np.int)
    plt.plot(n_vec, optimal_k_fitted, '--r')

    optimal_k = optimal_k_fitted

# get dnn error rate
measure = 'dnn_score'
dnn_error_rate = []
for root_dir in logdir_vec:
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data = json.load(f)
    dnn_error_rate.append(1.0 - data['test']['regular'][measure]['values'][0])
dnn_error_rate = np.array(dnn_error_rate)
dnn_error_rate_min_bayes = dnn_error_rate - approx_bayes_error_rate

# # get knn error rate
# measure = 'norm_{}_knn_score'.format(NORM)
# knn_error_rate = []
# for i, root_dir in enumerate(logdir_vec):
#     json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
#     with open(json_file) as f:
#         data = json.load(f)
#     k = optimal_k[i]
#     m_str = 'knn_k_{}_{}'.format(k, measure)
#     knn_error_rate.append(1.0 - data['test']['regular'][m_str]['values'][0])
# knn_error_rate = np.array(knn_error_rate)
# knn_error_rate_min_bayes = knn_error_rate - approx_bayes_error_rate

# now we set different C* for every n
if NORM == 'L1':
    C_vec = 1e-4 * np.array([9.611, 6.989, 6.308, 6.192, 5.466, 5.391, 5.871, 5.16, 5.273, 5.379])
elif NORM == 'L2':
    C_vec = np.array([0.02066, 0.01482, 0.01367, 0.0141, 0.0122, 0.01281, 0.01319, 0.01183, 0.01197, 0.01156])
else:
    raise AssertionError("No such metric {}".format(NORM))


if C_FIT:
    # we need to fit C*
    if NORM == 'L1':
        C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.000565, 0.00055, 0.00053, 0.000525, 0.00052])
    elif NORM == 'L2':
        pass
        # C_vec_fitted = np.array([C_vec[0], C_vec[1], C_vec[2], 0.0006, 0.000578, 0.000565, 0.00055, 0.00053, 0.000525, 0.00052])
    else:
        raise AssertionError('No such norm {}'.format(NORM))

    # plotting the C_vec and its fitted version:
    # plt.figure()
    # plt.plot(n_vec, C_vec, 'b')
    # plt.plot(n_vec, C_vec_fitted, '--r')
    # plt.title('C* vs number of training samples')
    # plt.ylabel('C*')
    # plt.xlabel('Number of training samples')
    # plt.legend(['from scatter', 'fitted'])
    # plt.grid()
    # plt.show()

    C_vec = C_vec_fitted

C1 = None  # need to be fitted.
bound = []

if C1_FIT == 'n_1':
    print('fit to just n=1')
    for i in range(0, len(n_vec)):
        n = n_vec[i]
        k_opt = optimal_k[i]
        dnn_err = dnn_error_rate_min_bayes[i]
        C = C_vec[i]
        if i == 0:
            part_bound = NOM / np.sqrt(k_opt) + np.exp(-3 * k_opt / 14) # + C1 * C * (k_opt / n)
            gap = dnn_err - part_bound
            C1 = gap / (C * k_opt / n)
            print('fitted C1={}'.format(C1))
        b = NOM / np.sqrt(k_opt) + np.exp(-3 * k_opt / 14) + C1 * C * (k_opt / n)
        bound.append(b)
else:
    print('fit to all n=1k,...,10k')
    C1_vec    = np.arange(0, 100, 0.001)
    error_vec = []
    for C1 in C1_vec:
        bound_tmp = []
        for i in range(0, len(n_vec)):
            n = n_vec[i]
            k_opt = optimal_k[i]
            C = C_vec[i]
            b = NOM / np.sqrt(k_opt) + np.exp(-3*k_opt/14) + C1*C*(k_opt/n)
            bound_tmp.append(b)
        bound_tmp = np.array(bound_tmp)
        diff = dnn_error_rate_min_bayes - bound_tmp
        error_vec.append(np.linalg.norm(diff))

    ind = np.argmin(error_vec)
    C1 = C1_vec[ind]
    print('fitted C1={}'.format(C1))

    for i in range(0, len(n_vec)):
        n = n_vec[i]
        k_opt = optimal_k[i]
        C = C_vec[i]
        b = NOM / np.sqrt(k_opt) + np.exp(-3 * k_opt / 14) + C1 * C * (k_opt / n)
        bound.append(b)

plt.figure()
plt.plot(n_vec, dnn_error_rate_min_bayes, 'k')
# plt.plot(n_vec, knn_error_rate_min_bayes, 'r')
plt.plot(n_vec, bound, 'b')
plt.title('error rates & bound: norm={}, C1_FIT={}, C1={}'.format(NORM, C1_FIT, C1))
# plt.legend(['DNN error rate', 'kNN error rate', 'Bound(C*, C1) (right term)'])
plt.legend(['DNN error rate', 'Bound(C*, C1) (right term)'])
plt.show()


# L2, C1=1.164
# L1, C1=31.096