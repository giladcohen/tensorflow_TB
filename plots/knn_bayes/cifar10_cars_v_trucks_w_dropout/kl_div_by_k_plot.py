"""
Plots for a single network (n=<something>) the D_KL(knn||DNN) vs k graph.
"""

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

num_classes = 2

n      = 10000
max_k  = n // num_classes
logdir = '/data/gilad/logs/knn_bayes/wrn/cifar10_cars_v_trucks/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=21011900'.format(n//1000)

json_file = os.path.join(logdir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)

measure = 'norm_{}_knn_kl_div_avg'.format(NORM)
k_vec      = []
kl_div_vec = []
for k in all_ks:
    if k <= max_k:
        m_str = 'knn_k_{}_{}'.format(k, measure)
        kl_div = data['test']['regular'][m_str]['values'][0]
        k_vec.append(k)
        kl_div_vec.append(kl_div)

fig = plt.figure(figsize=(8.0, 5.0))
ax2 = fig.add_subplot(111)
ax2.yaxis.grid()
ax2.set_ylabel('$D_{KL}(kNN||DNN)$', labelpad=5, fontdict={'fontsize': 12})
ax2.set_xlabel('k')
ax2.plot(k_vec, kl_div_vec, 'ko')
ax2.set_title('norm: {}. n={}'.format(NORM, n))
ax2.set_ylim(0.005, 0.015)

plt.tight_layout()
plt.savefig('kl_div_vs_k_n_{}k.png'.format(n//1000))
