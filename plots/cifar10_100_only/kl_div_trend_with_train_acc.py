"""Plotting the 9 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
subpos = np.array([0.35, 0.25, 0.5, 0.4])
fig = plt.figure(figsize=(10.0, 8.0))

# wrn, random_cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax4 = fig.add_subplot(321)
ax4.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax4.set_ylim(bottom=0, top=110)
ax4.set_title('Random CIFAR-10', fontdict={'fontsize': 18})
ax4.yaxis.grid()
ax4.tick_params(labelsize=14)
ax4.get_xaxis().set_visible(False)
ax4.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 18})

ax5 = fig.add_subplot(323)
ax5.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax5.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax5.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax5.yaxis.grid()
ax5.tick_params(labelsize=14)
ax5.get_xaxis().set_visible(False)
ax5.set_ylabel('$D_{KL}$ (train)', labelpad=5, fontdict={'fontsize': 18})
ax5.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'], loc=(0.57, 0.55), prop={'size': 14})

ax6 = fig.add_subplot(325)
ax6.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax6.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax6.plot(data['test']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax6.yaxis.grid()
ax6.tick_params(labelsize=14)
ax6.set_ylabel('$D_{KL}$ (test)', labelpad=5, fontdict={'fontsize': 18})
ax6.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'], loc=(0.57, 0.55), prop={'size': 14})

# wrn, random_cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax7 = fig.add_subplot(322)
ax7.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax7.set_ylim(bottom=0, top=110)
ax7.set_title('Random CIFAR-100', fontdict={'fontsize':18})
ax7.yaxis.grid()
ax7.get_xaxis().set_visible(False)
ax7.tick_params(labelsize=14)

ax8 = fig.add_subplot(324)
ax8.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax8.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax8.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax8.yaxis.grid()
ax8.tick_params(labelsize=14)
ax8.get_xaxis().set_visible(False)
ax8.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'], loc=(0.57, 0.65), prop={'size': 14})
subax8 = add_subplot_axes(ax8, subpos + [0.2, -0.0, 0, 0])
subax8.set_ylim([1.1, 1.2])
subax8.set_yticks([1.1, 1.2])
subax8.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][-11:], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][-11:]], 'r')
subax8.plot(data['train']['regular']['svm_kl_div2_avg']['steps'][-11:], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values'][-11:]], 'b')
subax8.plot(data['train']['regular']['lr_kl_div2_avg']['steps'][-11:] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values'][-11:]] , 'g')
subax8.tick_params(labelsize=12)
ax8.add_patch(patches.Polygon(xy=np.array([[40000, 5], [21300, 38], [45500, 92], [50000, 5]]), closed=True, color='silver'))
ax8.add_patch(patches.Rectangle(xy=(40000, -5), width=10000, height=10, facecolor='moccasin'))

ax9 = fig.add_subplot(326)
ax9.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax9.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax9.plot(data['test']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax9.yaxis.grid()
ax9.tick_params(labelsize=14)
ax9.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'], loc=(0.57, 0.55), prop={'size': 14})


plt.tight_layout()
plt.savefig('kl_div_trend_with_train_acc.png', dpi=350)