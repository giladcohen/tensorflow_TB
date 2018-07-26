from utils.plots import load_data_from_csv_wrapper
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

fig = plt.figure(figsize=(15.0, 8.0))

# wrn, cifar10
csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___ma_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = [steps[0]]  + steps[10:]
values = [values[0]] + values[10:]
ax2 = fig.add_subplot(111)
ax2.plot(steps, values, 'b')
ax2.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
ax2.set_xticklabels(['0', '10', '20', '30', '40', '50'], fontdict={'fontsize': 13})
ax2.set_xlabel('Thousands of train steps', fontdict={'fontsize': 13})
ax2.tick_params('y', colors='b')
ax2.yaxis.grid()
ax2.set_ylabel('MC score', color='b', labelpad=10, fontdict={'fontsize': 14})
ax2.set_title('CIFAR-10')
for item in ax2.get_yticklabels():
    item.set_fontsize(13)

csv_file = '/data/gilad/logs/ma_scores/wrn/cifar10/log_1444_070318_wrn_ma_score_wd_0.00078-SUPERSEED=07031800/data_for_figures/test___md_score'
steps, values = load_data_from_csv_wrapper(csv_file, mult=1.0)
steps  = [steps[0]]  + steps[10:]
values = [values[0]] + values[10:]
ax22 = ax2.twinx()
ax22.plot(steps, values, 'r')
ax22.set_ylim(bottom=0, top=1.045)
# ax22.yaxis.grid()
ax22.tick_params('y', colors='r')
ax22.set_ylabel('ME score', color='r', labelpad=10, fontdict={'fontsize': 14})
for item in ax22.get_yticklabels():
    item.set_fontsize(13)

# plt.subplots_adjust(wspace=0.25)
fig.tight_layout()
plt.show()
plt.savefig('ma_md_scores_just_cifar10_vs_iter.png', dpi=350)

