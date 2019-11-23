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
home_dir = '/data/gilad/logs/influence'
old_dir = {
    'cifar10' : os.path.join(home_dir, 'cifar10', 'log_080419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000'),
    'cifar100': os.path.join(home_dir, 'cifar100','log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01'),
    'svhn'    : os.path.join(home_dir, 'svhn_mini','log_300519_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_exp1')
}
new_dir = {
    'cifar10' : os.path.join(home_dir, 'cifar10', 'trained_model'),
    'cifar100': os.path.join(home_dir, 'cifar100', 'trained_model'),
    'svhn'    : os.path.join(home_dir, 'svhn', 'trained_model')
}

for dataset in ['cifar10', 'cifar100', 'svhn']:
    for subset in ['val', 'test']:
        if subset == 'val':
            indices = np.load(os.path.join(old_dir[dataset], 'val_indices.npy'))
        else:
            indices = range(1000)

        print('working on dataset {} for subset {}...'.format(dataset, subset))
        old_dir_tmp = old_dir[dataset]
        new_dir_tmp = new_dir[dataset]
        for i in tqdm(indices):
            val_dir = os.path.join(old_dir_tmp, subset, '{}_index_{}'.format(subset, i))
            real_path = os.path.join(val_dir, 'real')
            pred_path = os.path.join(val_dir, 'pred')
            adv_path  = os.path.join(val_dir, 'adv')

            new_pred  = os.path.join(new_dir_tmp, subset, '{}_index_{}'.format(subset, i), 'pred')
            new_adv   = os.path.join(new_dir_tmp, subset, '{}_index_{}'.format(subset, i), 'adv')

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





