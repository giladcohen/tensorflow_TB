from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import subprocess
import os
import numpy as np
from tqdm import tqdm

def run_cmd(cmd):
    # print ('start running command {} for i={}'.format(cmd, i))
    process = subprocess.call(cmd, shell=True)
    # print ('finished running command {} for i={}'.format(cmd, i))

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000
home_dir = '/data/gilad/logs/influence/cifar10'
old_dir = os.path.join(home_dir, 'log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000')
new_dir = os.path.join(home_dir, 'trained_model')

cifar10_val_inds  = np.load(os.path.join(old_dir, 'val_indices.npy'))
# cifar100_val_inds = np.load('/data/gilad/logs/influence/cifar100/log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01/val_indices.npy')
# svhn_mini_inds    = np.load('/data/gilad/logs/influence/svhn_mini/log_300519_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_exp1/val_indices.npy')

for i in tqdm(cifar10_val_inds):
    val_dir = os.path.join(old_dir, 'val', 'val_index_{}'.format(i))
    real_path = os.path.join(val_dir, 'real')
    pred_path = os.path.join(val_dir, 'pred')
    adv_path  = os.path.join(val_dir, 'adv')

    new_pred  = os.path.join(new_dir, 'val', 'val_index_{}'.format(i), 'pred')
    new_adv   = os.path.join(new_dir, 'val', 'val_index_{}'.format(i), 'adv')

    mkdir(new_pred)

    # copy pred
    if os.path.exists(pred_path):
        run_cmd('cp {} {}'.format(os.path.join(pred_path, 'scores.npy'), os.path.join(new_pred, 'scores.npy')))
    else:
        run_cmd('cp {} {}'.format(os.path.join(real_path, 'scores.npy'), os.path.join(new_pred, 'scores.npy')))

    # copy adv
    for attack in ['deepfool', 'jsma', 'cw', 'cw_nnif', 'fgsm']:
        scores_path = os.path.join(adv_path, attack, 'scores.npy')
        new_attack  = os.path.join(new_adv, attack)
        mkdir(new_attack)
        scores_path_new = os.path.join(new_attack, 'scores.npy')
        run_cmd('cp {} {}'.format(scores_path, scores_path_new))



