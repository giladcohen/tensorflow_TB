from utils.plots import load_data_from_csv_wrapper, add_subplot_axes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import json

plt.rcParams['interactive'] = False
fig = plt.figure(figsize=(10.0, 10.0))

USE_AVG = True
if not USE_AVG:
    sim_idx = 9

def average_sim_values(data_vec, dataset, key):
    """
    :param data_vec: vector of values to average
    :param dataset: 'train' or 'test'
    :param key: value to fetch
    :return: numpy array - average value over simulations
    """
    val_concat = \
        np.concatenate(
            (np.asanyarray(data_vec[0][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[1][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[2][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[3][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[4][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[5][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[6][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[7][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[8][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1),
             np.asanyarray(data_vec[9][dataset]['regular'][key]['values'], dtype=np.float64).reshape(1, -1)),
            axis=0)
    val_avg = np.average(val_concat, axis=0)
    return val_avg

# wrn, cifar10
# train accuracy - no multi
last_step = 15000
root_dir_vec = [
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111800',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111801',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111802',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111803',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111804',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111805',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111806',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111807',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111808',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111809'
]
data_single_sf_vec = []
for root_dir in root_dir_vec:
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data_single_sf_vec.append(json.load(f))
root_dir_vec = [
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111800',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111801',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111802',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111803',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111804',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111805',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111806',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111807',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111808',
    '/data/gilad/logs/multi_sf/cifar10/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111809'
]
data_multi_sf_vec = []
for root_dir in root_dir_vec:
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data_multi_sf_vec.append(json.load(f))

ax5 = fig.add_subplot(421)
steps                     = data_single_sf_vec[0]['train']['regular']['dnn_score']['steps']
last_idx = steps.index(last_step)
steps = steps[:last_idx+1]
if USE_AVG:
    dnn_score_single_sf = average_sim_values(data_single_sf_vec, 'train', 'dnn_score')[:last_idx+1]
    dnn_score_multi_sf  = average_sim_values(data_multi_sf_vec , 'train', 'dnn_score')[:last_idx+1]
else:
    dnn_score_single_sf = np.asanyarray(data_single_sf_vec[sim_idx]['train']['regular']['dnn_score']['values'], dtype=np.float64)[:last_idx+1]
    dnn_score_multi_sf  = np.asanyarray(data_multi_sf_vec[sim_idx]['train']['regular']['dnn_score']['values'], dtype=np.float64)[:last_idx+1]
ax5.plot(steps, 100 * dnn_score_single_sf, 'k')
ax5.plot(steps, 100 * dnn_score_multi_sf , 'r')
ax5.set_xticks([0, 5000, 10000, 15000])
ax5.tick_params(labelsize=14)
ax5.get_xaxis().set_visible(False)
ax5.set_ylabel('accuracy (%)', labelpad=5, fontdict={'fontsize': 18})
ax5.set_ylim(bottom=0, top=110)
ax5.set_title('CIFAR-10', fontdict={'fontsize': 18})
ax5.yaxis.grid()
ax5.legend(['single softmax', 'multi softmax'], loc=(0.25, 0.03), prop={'size': 17})

ax6 = fig.add_subplot(423)
if USE_AVG:
    dnn_score_single_sf = average_sim_values(data_single_sf_vec, 'test', 'dnn_score')[:last_idx+1]
    dnn_score_multi_sf  = average_sim_values(data_multi_sf_vec , 'test', 'dnn_score')[:last_idx+1]
else:
    dnn_score_single_sf = np.asanyarray(data_single_sf_vec[sim_idx]['test']['regular']['dnn_score']['values'], dtype=np.float64)[:last_idx+1]
    dnn_score_multi_sf  = np.asanyarray(data_multi_sf_vec[sim_idx]['test']['regular']['dnn_score']['values'], dtype=np.float64)[:last_idx+1]
ax6.plot(steps, 100 * dnn_score_single_sf, 'k')
ax6.plot(steps, 100 * dnn_score_multi_sf , 'r')
ax6.set_xticks([0, 5000, 10000, 15000])
ax6.get_xaxis().set_visible(False)
ax6.tick_params(labelsize=14)
ax6.set_ylabel('accuracy (%)', labelpad=14, fontdict={'fontsize': 18})
ax6.yaxis.grid()
ax6.set_ylim(0, 80)
ax6.legend(['single softmax', 'multi softmax'], loc=(0.25, 0.03), prop={'size': 17})

ax7 = fig.add_subplot(425)
if USE_AVG:
    knn_kl_div2_avg_avg_train = average_sim_values(data_single_sf_vec, 'train', 'knn_kl_div2_avg')[:last_idx+1]
    knn_kl_div2_avg_avg_test  = average_sim_values(data_single_sf_vec, 'test', 'knn_kl_div2_avg')[:last_idx+1]
else:
    knn_kl_div2_avg_avg_train = np.asanyarray(data_single_sf_vec[sim_idx]['train']['regular']['knn_kl_div2_avg']['values'], dtype=np.float64)[:last_idx+1]
    knn_kl_div2_avg_avg_test  = np.asanyarray(data_single_sf_vec[sim_idx]['test']['regular']['knn_kl_div2_avg']['values'], dtype=np.float64)[:last_idx+1]
ax7.plot(steps, knn_kl_div2_avg_avg_train, 'k')
ax7.plot(steps, knn_kl_div2_avg_avg_test,  'k--')
ax7.set_xticks([0, 5000, 10000, 15000])
ax7.tick_params(labelsize=14)
ax7.get_xaxis().set_visible(False)
ax7.set_ylabel('$D_{KL}$($k$-NN || DNN)', labelpad=10, fontdict={'fontsize': 16})
ax7.set_ylim(-0.04, 1.2)
ax7.yaxis.grid()
ax7.legend(['train', 'test'], loc=(0.58, 0.45), prop={'size': 17})

ax8 = fig.add_subplot(427)
if USE_AVG:
    knn_kl_div2_avg_avg_train = average_sim_values(data_multi_sf_vec, 'train', 'knn_kl_div2_avg')[:last_idx+1]
    knn_kl_div2_avg_avg_test  = average_sim_values(data_multi_sf_vec, 'test', 'knn_kl_div2_avg')[:last_idx+1]
else:
    knn_kl_div2_avg_avg_train = np.asanyarray(data_multi_sf_vec[sim_idx]['train']['regular']['knn_kl_div2_avg']['values'], dtype=np.float64)[:last_idx+1]
    knn_kl_div2_avg_avg_test  = np.asanyarray(data_multi_sf_vec[sim_idx]['test']['regular']['knn_kl_div2_avg']['values'], dtype=np.float64)[:last_idx+1]
ax8.plot(steps, knn_kl_div2_avg_avg_train, 'r')
ax8.plot(steps, knn_kl_div2_avg_avg_test,  'r--')
ax8.set_xticks([0, 5000, 10000, 15000])
ax8.tick_params(labelsize=14)
ax8.set_ylabel('$D_{KL}$($k$-NN || DNN)', labelpad=10, fontdict={'fontsize': 16})
ax8.set_ylim(-0.04, 1.2)
ax8.yaxis.grid()
ax8.legend(['train', 'test'], loc=(0.58, 0.45), prop={'size': 17})

# wrn, cifar100
# train accuracy - no multi
last_step = 40000
root_dir_vec = [
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111800',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111801',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111802',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111803',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111804',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111805',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111806',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111807',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111808',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_no_multi_sf_lr_0.01-SUPERSEED=01111809'
]
data_single_sf_vec = []
for root_dir in root_dir_vec:
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data_single_sf_vec.append(json.load(f))
root_dir_vec = [
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111800',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111801',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111802',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111803',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111804',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111805',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111806',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111807',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111808',
    '/data/gilad/logs/multi_sf/cifar100/overfitting/log_1331_011118_1_3_0.125_2_3_0.25_3_2_0.5_lr_0.01-SUPERSEED=01111809'
]
data_multi_sf_vec = []
for root_dir in root_dir_vec:
    json_file = os.path.join(root_dir, 'data_for_figures', 'data.json')
    with open(json_file) as f:
        data_multi_sf_vec.append(json.load(f))

ax9 = fig.add_subplot(422)
steps                     = data_single_sf_vec[0]['train']['regular']['dnn_score']['steps']
last_idx = steps.index(last_step)
steps = steps[:last_idx+1]
if USE_AVG:
    dnn_score_single_sf = average_sim_values(data_single_sf_vec, 'train', 'dnn_score')[:last_idx+1]
    dnn_score_multi_sf  = average_sim_values(data_multi_sf_vec , 'train', 'dnn_score')[:last_idx+1]
else:
    dnn_score_single_sf = np.asanyarray(data_single_sf_vec[sim_idx]['train']['regular']['dnn_score']['values'], dtype=np.float64)[:last_idx+1]
    dnn_score_multi_sf  = np.asanyarray(data_multi_sf_vec[sim_idx]['train']['regular']['dnn_score']['values'], dtype=np.float64)[:last_idx+1]
ax9.plot(steps, 100 * dnn_score_single_sf, 'k')
ax9.plot(steps, 100 * dnn_score_multi_sf , 'r')
ax9.set_xticks([0, 10000, 20000, 30000, 40000])
ax9.tick_params(labelsize=14)
ax9.get_xaxis().set_visible(False)
ax9.set_ylim(bottom=0, top=110)
ax9.set_title('CIFAR-100', fontdict={'fontsize': 18})
ax9.yaxis.grid()
ax9.legend(['single softmax', 'multi softmax'], loc=(0.25, 0.03), prop={'size': 17})

ax10 = fig.add_subplot(424)
if USE_AVG:
    dnn_score_single_sf = average_sim_values(data_single_sf_vec, 'test', 'dnn_score')[:last_idx+1]
    dnn_score_multi_sf  = average_sim_values(data_multi_sf_vec , 'test', 'dnn_score')[:last_idx+1]
else:
    dnn_score_single_sf = np.asanyarray(data_single_sf_vec[sim_idx]['test']['regular']['dnn_score']['values'], dtype=np.float64)[:last_idx+1]
    dnn_score_multi_sf  = np.asanyarray(data_multi_sf_vec[sim_idx]['test']['regular']['dnn_score']['values'], dtype=np.float64)[:last_idx+1]
ax10.plot(steps, 100 * dnn_score_single_sf, 'k')
ax10.plot(steps, 100 * dnn_score_multi_sf , 'r')
ax10.set_xticks([0, 10000, 20000, 30000, 40000])
ax10.tick_params(labelsize=14)
ax10.get_xaxis().set_visible(False)
ax10.yaxis.grid()
ax10.set_ylim(0, 30)
ax10.legend(['single softmax', 'multi softmax'], loc=(0.25, 0.03), prop={'size': 17})

ax11 = fig.add_subplot(426)
if USE_AVG:
    knn_kl_div2_avg_avg_train = average_sim_values(data_single_sf_vec, 'train', 'knn_kl_div2_avg')[:last_idx+1]
    knn_kl_div2_avg_avg_test  = average_sim_values(data_single_sf_vec, 'test', 'knn_kl_div2_avg')[:last_idx+1]
else:
    knn_kl_div2_avg_avg_train = np.asanyarray(data_single_sf_vec[sim_idx]['train']['regular']['knn_kl_div2_avg']['values'], dtype=np.float64)[:last_idx+1]
    knn_kl_div2_avg_avg_test  = np.asanyarray(data_single_sf_vec[sim_idx]['test']['regular']['knn_kl_div2_avg']['values'], dtype=np.float64)[:last_idx+1]
ax11.plot(steps, knn_kl_div2_avg_avg_train, 'k')
ax11.plot(steps, knn_kl_div2_avg_avg_test,  'k--')
ax11.set_xticks([0, 10000, 20000, 30000, 40000])
ax11.tick_params(labelsize=14)
ax11.get_xaxis().set_visible(False)
ax11.set_ylim(-0.04, 3.2)
ax11.yaxis.grid()
ax11.legend(['train', 'test'], loc=(0.58, 0.45), prop={'size': 17})

ax12 = fig.add_subplot(428)
if USE_AVG:
    knn_kl_div2_avg_avg_train = average_sim_values(data_multi_sf_vec, 'train', 'knn_kl_div2_avg')[:last_idx+1]
    knn_kl_div2_avg_avg_test  = average_sim_values(data_multi_sf_vec, 'test', 'knn_kl_div2_avg')[:last_idx+1]
else:
    knn_kl_div2_avg_avg_train = np.asanyarray(data_multi_sf_vec[sim_idx]['train']['regular']['knn_kl_div2_avg']['values'], dtype=np.float64)[:last_idx+1]
    knn_kl_div2_avg_avg_test  = np.asanyarray(data_multi_sf_vec[sim_idx]['test']['regular']['knn_kl_div2_avg']['values'], dtype=np.float64)[:last_idx+1]
ax12.plot(steps, knn_kl_div2_avg_avg_train, 'r')
ax12.plot(steps, knn_kl_div2_avg_avg_test,  'r--')
ax12.set_xticks([0, 10000, 20000, 30000, 40000])
ax12.tick_params(labelsize=14)
ax12.set_ylim(-0.04, 3.2)
ax12.yaxis.grid()
ax12.legend(['train', 'test'], loc=(0.58, 0.45), prop={'size': 17})

# plt.tight_layout()
plt.subplots_adjust(hspace=0.17)
if USE_AVG:
    plt.savefig('dl_div_overfitting_multi_sf.png', dpi=350)
else:
    plt.savefig('dl_div_overfitting_multi_sf_sim_{}.png'.format(sim_idx), dpi=350)
