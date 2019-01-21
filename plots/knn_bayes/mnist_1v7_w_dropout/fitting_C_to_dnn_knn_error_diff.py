from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

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

NORM   = 'L2'
C1_FIT = 'n_4up'  # n_1 or n_4up n_all
K_FIT = True
NOM = 2.426

approx_bayes_error_rate = 0.0014

logdir_vec  = []
n_vec       = []
max_ks      = []
num_classes = 2
for i in range(1, 13):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/mnist_1v7/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=08011900'.format(i))
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
    # override:
    if NORM == 'L2':
        optimal_k[0] = 15
        optimal_k[1] = 50
        optimal_k[2] = 85

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

# get knn error rate
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
    #C_vec = 1e-4 * np.array([2.083, 2.25, 3, 2.5, 2.9593, 4.3832, 2.4873, 3.6102, 3.9432, 3.8414, 1.8197, 1.9476])
    C_vec = 1e-4 * np.array([2.165, 2.225, 2.5, 2.225, 3.0, 4.39, 2.5, 3.658, 3.992, 4.006, 1.8, 1.7])
elif NORM == 'L2':
    #C_vec = 1e-3 * np.array([4.3571, 8.4932, 6.6146, 5.9205, 6.581, 9.1681, 5.1421, 7.5114, 8.5654, 8.3273, 3.549, 4.058])
    C_vec = 1e-3 * np.array([4.388, 4.2, 5, 4.5, 6, 9.24, 5.2, 7.71, 8.68, 8.5, 3.7, 3.5])
else:
    raise AssertionError("No such metric {}".format(NORM))

# for mnist_1v7 we need to fit C* to a constant
C_constant = 1e-4 * np.arange(2, 90, 0.1)
error_vec = []
for C in C_constant:
    C_vec_min_const = C_vec - C
    diff = np.linalg.norm(C_vec_min_const)
    error_vec.append(diff)

ind = np.argmin(error_vec)
C = C_constant[ind]
print('fitted C={}'.format(C))

# override:
C_vec = np.array([0.00022, 0.00022, 0.00022, 0.00022, \
              0.0003, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, \
              0.00015, 0.00015])

# printing:
plt.figure()
plt.plot(n_vec, C_vec),
plt.plot(n_vec, C*np.ones(12), '--r')
plt.ylabel('C*')
plt.xlabel('n')

C1 = 0
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
elif C1_FIT == 'n_all':
    print('fit to all n=1k,...,12k')
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
else:
    print('fit to n=4k,...,12k')
    C1_vec    = np.arange(0, 100, 0.001)
    error_vec = []
    for C1 in C1_vec:
        bound_tmp = []
        for i in range(3, len(n_vec)):
            n = n_vec[i]
            k_opt = optimal_k[i]
            C = C_vec[i]
            b = NOM / np.sqrt(k_opt) + np.exp(-3*k_opt/14) + C1*C*(k_opt/n)
            bound_tmp.append(b)
        bound_tmp = np.array(bound_tmp)
        diff = dnn_error_rate_min_bayes[3:] - bound_tmp
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
plt.plot(n_vec[3:], dnn_error_rate_min_bayes[3:], 'k')
# plt.plot(n_vec[3:], knn_error_rate_min_bayes[3:], 'r')
plt.plot(n_vec[3:], bound[3:], 'b')
plt.title('DNN and kNN error rates and bound for norm={} and C1_FIT={}'.format(NORM, C1_FIT))
plt.legend(['DNN error rate', 'kNN error rate', 'Bound(C*, C1) (right term)'])
plt.show()


# L2, n_all, C1=0
# L1, n_all, C1=0
# plt.figure()
# plt.plot(n_vec, C_vec)
# plt.ylabel('C*')
# plt.xlabel('n')
# plt.title('C* as a function of n for norm=L2 for cifar10_cats_v_dogs')
# plt.show()