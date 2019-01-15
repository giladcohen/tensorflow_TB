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

logdir_vec = []
n_vec      = []
max_ks     = []
for i in range(1, 16):
    if i in [4, 5]:  # not ready yet
        continue
    logdir_vec.append('/data/gilad/logs/knn_bayes/wrn/cifar10/w_dropout/log_bs_200_lr_0.1s_n_{}k-SUPERSEED=30121800'.format(i))
    n_vec.append(int(i * 1000))
    max_ks.append(int(i * 100))

measure = 'norm_L2_knn_kl_div2_avg'
knn_score = []
optimal_k      = []
for i, root_dir in enumerate(logdir_vec):
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    max_k = max_ks[i]
    with open(json_file) as f:
        data = json.load(f)
    best_score = np.inf  # lower is better
    best_k     = None
    for k in all_ks:
        if k <= max_k:
            m_str = 'knn_k_{}_{}'.format(k, measure)
            score = data['test']['regular'][m_str]['values'][0]
            if score < best_score:
                best_score = score
                best_k = k
    knn_score.append(best_score)
    optimal_k.append(best_k)

ax1 = fig.add_subplot(211)
ax1.plot(n_vec, knn_score)
ax1.yaxis.grid()
ax1.set_ylabel('k-NN score', labelpad=5, fontdict={'fontsize': 12})
ax1.set_xlabel('number of samples')
ax1.set_title('k-NN score for optimal K. measure: {}'.format(measure))

ax2 = fig.add_subplot(212)
ax2.plot(n_vec, optimal_k)
ax2.yaxis.grid()
ax2.set_ylabel('optimal k', labelpad=5, fontdict={'fontsize': 12})
ax2.set_xlabel('number of samples')
# ax2.set_ylim(top=100)
ax2.set_title('optimal K (max knn_score)')

plt.tight_layout()
plt.savefig('knn_score_optimal_k_all_measures.png')

