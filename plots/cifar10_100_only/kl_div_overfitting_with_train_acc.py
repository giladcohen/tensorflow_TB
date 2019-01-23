from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(10.0, 8.0))

# wrn, cifar-10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/overfitting/log_0311_270918_metrics-SUPERSEED=27091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)

ax4 = fig.add_subplot(321)
ax4.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax4.set_ylim(bottom=0, top=105)
ax4.set_title('CIFAR-10', fontdict={'fontsize': 18})
ax4.yaxis.grid()
ax4.set_xticks([0, 5000, 10000, 15000, 20000])
ax4.set_ylabel('train accuracy (%)', labelpad=7, fontdict={'fontsize': 18})
ax4.get_xaxis().set_visible(False)
ax4.tick_params(labelsize=14)

ax5 = fig.add_subplot(323)
# ax5.plot(1000, 67.34, 'ro')
ax5.plot(data['test']['regular']['dnn_score']['steps'], [v * 100 for v in data['test']['regular']['dnn_score']['values']], 'k')
ax5.set_ylim(bottom=0, top=110)
ax5.yaxis.grid()
ax5.set_ylim(60, 70)
ax5.set_xticks([0, 5000, 10000, 15000, 20000])
ax5.set_ylabel('test accuracy (%)', labelpad=15, fontdict={'fontsize': 18})
ax5.get_xaxis().set_visible(False)
ax5.tick_params(labelsize=14)

ax6 = fig.add_subplot(325)
ax6.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'k')
ax6.plot(data['test']['regular']['knn_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']] , 'k--')
# ax6.set_ylim(top=30)
ax6.set_xticks([0, 5000, 10000, 15000, 20000])
ax6.yaxis.grid()
ax6.legend(['train', 'test'], loc=(0.65, 0.19), prop={'size': 16})
ax6.set_ylabel('$D_{KL}$($k$-NN || DNN)', labelpad=14, fontdict={'fontsize': 17})
ax6.tick_params(labelsize=14)

# wrn, cifar-100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/overfitting/log_2046_061018_metrics-SUPERSEED=06101803'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)

ax7 = fig.add_subplot(322)
ax7.plot(data['train']['regular']['dnn_score']['steps'][:74], [v * 100 for v in data['train']['regular']['dnn_score']['values'][0:74]], 'k')
ax7.set_ylim(bottom=0, top=105)
ax7.set_title('CIFAR-100', fontdict={'fontsize': 18})
ax7.yaxis.grid()
ax7.set_xticks([0, 5000, 10000, 15000])
ax7.get_xaxis().set_visible(False)
ax7.tick_params(labelsize=14)

ax8 = fig.add_subplot(324)
# ax8.plot(3200, 29.85, 'ro')
ax8.plot(data['test']['regular']['dnn_score']['steps'][:74], [v * 100 for v in data['test']['regular']['dnn_score']['values'][0:74]], 'k')
ax8.set_ylim(bottom=10, top=33)
ax8.yaxis.grid()
ax8.set_xticks([0, 5000, 10000, 15000])
ax8.get_xaxis().set_visible(False)
ax8.tick_params(labelsize=14)

ax9 = fig.add_subplot(326)
ax9.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][:74], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][:74]], 'k')
ax9.plot(data['test']['regular']['knn_kl_div2_avg']['steps'][:74] , [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values'][:74]] , 'k--')
# ax9.set_ylim(top=30)
ax9.set_xticks([0, 5000, 10000, 15000])
ax9.yaxis.grid()
ax9.legend(['train', 'test'], loc=(0.65, 0.45), prop={'size': 16})
ax9.tick_params(labelsize=14)

plt.tight_layout()
plt.savefig('kl_div_overfitting_with_train_acc.png', dpi=350)