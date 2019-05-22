import subprocess
import time
import os

def run_cmd(cmd, i):
    print ('start running command {} for i={}'.format(cmd, i))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {} for i={}'.format(cmd, i))

# dump_file = '/data/gilad/logs/influence/cifar100/log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01/test/dump.txt'
# with open(dump_file) as f:
#     content = f.readlines()
# # # you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content]
# del content[0]
#
# dir_list = []
# for c in content:
#     dir_list.append(os.path.join('/data/gilad/logs/influence/cifar100/log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01/test', c))
#
# for dir in dir_list:
#     cmd = 'mkdir -p ' + os.path.join(dir, 'adv', 'deepfool')
#     run_cmd(cmd)
#     cmd = 'mv ' + os.path.join(dir, 'adv', '*') + ' ' + os.path.join(dir, 'adv', 'deepfool')
#     run_cmd(cmd)

home_dir = '/data/gilad/logs/influence/cifar100/log_300419_b_125_wd_0.0004_mom_lr_0.1_f_0.9_p_3_c_2_val_size_1000_ls_0.01/test'
for i in range(7, 10000):
    adv_dir = os.path.join(home_dir, 'test_index_{}'.format(i), 'adv')
    deepfool_dir = os.path.join(adv_dir, 'deepfool')
    if not os.path.exists(deepfool_dir):  # path not exist
        run_cmd('mkdir -p ' + deepfool_dir, i)
        # go to dir
        os.chdir(adv_dir)
        run_cmd('mv `find . -type f | grep -v "/cw/"` deepfool', i)
    else:  # path exist
        # check if empty
        if len(os.listdir(deepfool_dir)) == 0:  # indeed empty
            os.chdir(adv_dir)
            run_cmd('mv `find . -type f | grep -v "/cw/"` deepfool', i)


