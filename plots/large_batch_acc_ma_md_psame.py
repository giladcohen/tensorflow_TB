"""Plotting the accuracy of the large batch experiment, following MA/MD and P_SAME"""
from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

subpos = [0.4, 0.5, 0.48, 0.3]
fig = plt.figure(figsize=(14, 4))


# mnist
# fc2net

# accuracies
csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___score'
steps, acc_test_values = load_data_from_csv_wrapper(csv_file, mult=100.0, round_points=4)
# csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___score_trainset'
# steps, acc_train_values = load_data_from_csv_wrapper(csv_file, mult=100.0, round_points=4)
csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___knn_score'
steps, acc_knn_test_values = load_data_from_csv_wrapper(csv_file, mult=100.0, round_points=4)
# csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___knn_score_trainset'
# steps, acc_knn_train_values = load_data_from_csv_wrapper(csv_file, mult=100.0, round_points=4)
ax1 = fig.add_subplot(131)
ax1.plot(steps[0:50], acc_test_values[0:50]     , 'k')
# ax1.plot(steps[0:50], acc_train_values[0:50]    , 'k--')
ax1.plot(steps[0:50], acc_knn_test_values[0:50] , 'r')
# ax1.plot(steps[0:50], acc_knn_train_values[0:50], 'r--')
ax1.set_ylim(bottom=91, top=99)
ax1.set_ylabel('accuracy', color='k', labelpad=5, fontdict={'fontsize': 14})
ax1.yaxis.grid()
ax1.legend(['DNN', 'k-NN'], loc=(0.62, 0.62), prop={'size': 14})
for item in ax1.get_xticklabels():
    item.set_fontsize(13)
for item in ax1.get_yticklabels():
    item.set_fontsize(13)

# MA and MD scores
csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___ma_score'
steps, ma_test_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=4)
csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___md_score'
steps, md_test_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=4)
steps          = [steps[0]] + steps[10:50]
ma_test_values = [ma_test_values[0]] + ma_test_values[10:50]
md_test_values = [md_test_values[0]] + md_test_values[10:50]
ax2 = fig.add_subplot(132)
ax2.plot(steps, ma_test_values, 'b')
ax2.set_ylim(bottom=0.95, top=1)
ax2.set_ylabel('$MC$ score', color='b', labelpad=5, fontdict={'fontsize': 14})
ax2.tick_params('y', colors='b')
ax2.yaxis.grid()
ax22 = ax2.twinx()
ax22.plot(steps, md_test_values, 'r')
ax22.set_ylim(bottom=0, top=1)
ax22.tick_params('y', colors='r')
ax22.set_ylabel('$ME$ score', color='r', labelpad=5, fontdict={'fontsize': 14})
for item in ax2.get_xticklabels():
    item.set_fontsize(13)
for item in ax2.get_yticklabels():
    item.set_fontsize(13)
for item in ax22.get_yticklabels():
    item.set_fontsize(13)

# P_SAME score
csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___score'
steps, acc_test_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___ma_score'
steps, ma_test_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
csv_file = '/data/gilad/logs/ma_scores/large_batch/fc2net/mnist/log_1053_080518_ma_score_w_trainset_fc2net_mnist_large_batch_no_aug_wd_0.0_lr_10_schedule_200_2_300_0.4_400_0.08-SUPERSEED=08051800/data_for_figures/test___md_score'
steps, md_test_values = load_data_from_csv_wrapper(csv_file, mult=1.0, round_points=8)
steps          = [steps[0]] + steps[10:50]
ma_test_values = [ma_test_values[0]] + ma_test_values[10:50]
md_test_values = [md_test_values[0]] + md_test_values[10:50]
P_SAME = [ma * acc + md * (1.0 - acc) for ma, md, acc in zip(ma_test_values, md_test_values, acc_test_values)]
P_SAME = [round(elem, 4) for elem in P_SAME]
ax3 = fig.add_subplot(133)
ax3.plot(steps, P_SAME, 'k')
ax3.set_ylabel('$P_{SAME}$', labelpad=3, fontdict={'fontsize': 15})
ax3.set_ylim(bottom=0.0, top=1.045)
ax3.yaxis.grid()
for item in ax3.get_xticklabels():
    item.set_fontsize(13)
for item in ax3.get_yticklabels():
    item.set_fontsize(13)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
fig.tight_layout()
plt.savefig('acc_ma_md_psame_large_batch.png', dpi=350)
