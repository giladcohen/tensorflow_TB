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

logdir_vec  = []
n_vec       = []
max_ks      = []
num_classes = 2
for i in range(1, 11):
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10_cats_v_dogs/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=08011900'.format(i))
    n_vec.append(int(i * 1000))
    max_ks.append(int(i * 1000 / num_classes))

measure = 'norm_L2_knn_kl_div2_avg'
# measure = 'norm_L2_knn_score'
knn_score = []
optimal_k = []
for i, root_dir in enumerate(logdir_vec):
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    max_k      = max_ks[i]
    best_score = np.inf  # lower is better
    best_k     = None
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

measure = 'norm_L2_knn_kl_div2_avg'
# measure = 'norm_L2_knn_score'
knn_score_bayesian = []
optimal_k_bayesian = []
for i, root_dir in enumerate(logdir_vec):
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    max_k      = max_ks[i]
    best_score = np.inf  # lower is better
    best_k     = None
    with open(json_file) as f:
        data = json.load(f)
    for k in all_ks:
        if k <= max_k:
            m_str = 'knn_dropout_0_5_k_{}_{}'.format(k, measure)
            score = data['test']['regular'][m_str]['values'][0]
            if score < best_score:
                best_score = score
                best_k = k
    knn_score_bayesian.append(best_score)
    optimal_k_bayesian.append(best_k)

# for just dnn error rate
dnn_error_rate = []
for i, root_dir in enumerate(logdir_vec):
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data = json.load(f)
    dnn_error_rate.append(1.0 - data['test']['regular']['dnn_score']['values'][0])
dnn_error_rate = np.array(dnn_error_rate)
# approx_bayes_error_rate = 0.0047
# dnn_error_rate_min_bayes = dnn_error_rate - approx_bayes_error_rate

ax1 = fig.add_subplot(211)
ax1.plot(n_vec, knn_score, 'r')
ax1.plot(n_vec, knn_score_bayesian, 'r--')
# ax1.plot(n_vec, dnn_error_rate, 'k')
ax1.yaxis.grid()
# ax1.set_ylabel('error rate', labelpad=5, fontdict={'fontsize': 12})
ax1.set_ylabel('$D_{KL}$', labelpad=5, fontdict={'fontsize': 12})
ax1.set_xlabel('number of samples')
# ax1.set_title('DNN and k-NN error rates for optimal K. measure: {}'.format(measure))
ax1.set_title('$D_{KL}$' + '(DNN || k-NN) for optimal K. measure: {}'.format(measure))
# ax1.legend(['k-NN', 'k-NN (Bayesian)', 'DNN'])
ax1.legend(['k-NN', 'k-NN (Bayesian)'])

ax2 = fig.add_subplot(212)
ax2.plot(n_vec, optimal_k, 'k')
ax2.plot(n_vec, optimal_k_bayesian, 'k--')
ax2.yaxis.grid()
ax2.set_ylabel('optimal k', labelpad=5, fontdict={'fontsize': 12})
ax2.set_xlabel('number of samples')
# ax2.set_ylim(top=100)
ax2.set_title('optimal K (max knn_score)')
ax2.legend(['k-NN', 'k-NN (Bayesian)'])

plt.tight_layout()
plt.savefig('knn_score_optimal_k.png')

