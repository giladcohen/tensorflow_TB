"""Plotting the 9 KNN accuracy plots"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
subpos = np.array([0.35, 0.25, 0.5, 0.4])
fig = plt.figure(figsize=(15.0, 8.0))

# wrn, random_mnist
root_dir = '/data/gilad/logs/metrics/wrn/mnist/random/log_0333_250918_metrics_longer-SUPERSEED=25091800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax1 = fig.add_subplot(331)
ax1.plot(data['train']['regular']['dnn_score']['steps'][:-10], [v * 100 for v in data['train']['regular']['dnn_score']['values'][:-10]], 'k')
ax1.set_ylim(bottom=0, top=110)
ax1.set_title('RANDOM MNIST')
ax1.yaxis.grid()
ax1.set_ylabel('accuracy (%)', labelpad=0.5, fontdict={'fontsize': 12})

ax2 = fig.add_subplot(334)
ax2.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][:-10], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][:-10]], 'r')
ax2.plot(data['train']['regular']['svm_kl_div2_avg']['steps'][:-10], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values'][:-10]], 'b')
ax2.plot(data['train']['regular']['lr_kl_div2_avg']['steps'][:-10] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values'][:-10]] , 'g')
ax2.set_ylabel('$D_{KL}$ (train)', labelpad=5, fontdict={'fontsize': 12})
ax2.yaxis.grid()
ax2.set_yticks([0, 50, 100, 150, 200])
ax2.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'])

ax3 = fig.add_subplot(337)
ax3.plot(data['test']['regular']['knn_kl_div2_avg']['steps'][:-10], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values'][:-10]], 'r')
ax3.plot(data['test']['regular']['svm_kl_div2_avg']['steps'][:-10], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values'][:-10]], 'b')
ax3.plot(data['test']['regular']['lr_kl_div2_avg']['steps'][:-10] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values'][:-10]] , 'g')
ax3.set_ylabel('$D_{KL}$ (test)', labelpad=5, fontdict={'fontsize': 12})
ax3.yaxis.grid()
ax3.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'])
subax3 = add_subplot_axes(ax3, subpos - [0, 0.21, 0, 0])
subax3.set_ylim([0, 20])
subax3.set_yticks([0, 20])
subax3.plot(data['test']['regular']['knn_kl_div2_avg']['steps'][:-10][-11:], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values'][:-10][-11:]], 'r')
subax3.plot(data['test']['regular']['svm_kl_div2_avg']['steps'][:-10][-11:], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values'][:-10][-11:]], 'b')
subax3.plot(data['test']['regular']['lr_kl_div2_avg']['steps'][:-10][-11:] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values'][:-10][-11:]] , 'g')
ax3.add_patch(patches.Polygon(xy=np.array([[40000, 20], [26500, 47], [48700, 120], [50000, 20]]), closed=True, color='silver'))
ax3.add_patch(patches.Rectangle(xy=(40000, 0), width=10000, height=20, facecolor='moccasin'))

# wrn, random_cifar10
root_dir = '/data/gilad/logs/metrics/wrn/cifar10/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax4 = fig.add_subplot(332)
ax4.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax4.set_ylim(bottom=0, top=110)
ax4.set_title('RANDOM CIFAR-10')
ax4.yaxis.grid()

ax5 = fig.add_subplot(335)
ax5.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax5.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax5.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax5.yaxis.grid()
ax5.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'])


ax6 = fig.add_subplot(338)
ax6.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax6.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax6.plot(data['test']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax6.yaxis.grid()
ax6.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'])

# wrn, random_cifar100
root_dir = '/data/gilad/logs/metrics/wrn/cifar100/random/log_1405_230818_metrics_w_confidence-SUPERSEED=23081800'
json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
with open(json_file) as f:
    data = json.load(f)
ax7 = fig.add_subplot(333)
ax7.plot(data['train']['regular']['dnn_score']['steps'], [v * 100 for v in data['train']['regular']['dnn_score']['values']], 'k')
ax7.set_ylim(bottom=0, top=110)
ax7.set_title('RANDOM CIFAR-100')
ax7.yaxis.grid()

ax8 = fig.add_subplot(336)
ax8.plot(data['train']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values']], 'r')
ax8.plot(data['train']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values']], 'b')
ax8.plot(data['train']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax8.yaxis.grid()
ax8.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'])
subax8 = add_subplot_axes(ax8, subpos + [0.4, 0.0, 0, 0])
subax8.set_ylim([1.1, 1.2])
subax8.set_yticks([1.1, 1.2])
subax8.plot(data['train']['regular']['knn_kl_div2_avg']['steps'][-11:], [v * 100 for v in data['train']['regular']['knn_kl_div2_avg']['values'][-11:]], 'r')
subax8.plot(data['train']['regular']['svm_kl_div2_avg']['steps'][-11:], [v * 100 for v in data['train']['regular']['svm_kl_div2_avg']['values'][-11:]], 'b')
subax8.plot(data['train']['regular']['lr_kl_div2_avg']['steps'][-11:] , [v * 100 for v in data['train']['regular']['lr_kl_div2_avg']['values'][-11:]] , 'g')
ax8.add_patch(patches.Polygon(xy=np.array([[40000, 5], [25500, 35], [47000, 92], [50000, 5]]), closed=True, color='silver'))
ax8.add_patch(patches.Rectangle(xy=(40000, -5), width=10000, height=10, facecolor='moccasin'))


ax9 = fig.add_subplot(339)
ax9.plot(data['test']['regular']['knn_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['knn_kl_div2_avg']['values']], 'r')
ax9.plot(data['test']['regular']['svm_kl_div2_avg']['steps'], [v * 100 for v in data['test']['regular']['svm_kl_div2_avg']['values']], 'b')
ax9.plot(data['test']['regular']['lr_kl_div2_avg']['steps'] , [v * 100 for v in data['test']['regular']['lr_kl_div2_avg']['values']] , 'g')
ax9.yaxis.grid()
ax9.legend(['$k$-NN||DNN', 'SVM||DNN', 'LR||DNN'])


plt.tight_layout()
plt.savefig('kl_div_trend_with_train_acc.png')