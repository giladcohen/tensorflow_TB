import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

norm_vec       = ['L1', 'L2']
percentage_vec = ['0.5', '10', '100']
input_vec      = ['image', 'embedding']

for NORM in norm_vec:
    for PERCENTAGE in percentage_vec:
        for INPUT in input_vec:
            cmd = 'python plots/knn_bayes/cifar10_cats_v_dogs_w_dropout/calc_lipschits_constant.py' + \
                  ' --NORM ' + NORM + \
                  ' --PERCENTAGE ' + PERCENTAGE + \
                  ' --INPUT ' + INPUT + \
                  ' --DATASET_NAME cifar10_cats_v_dogs'
            run_cmd(cmd)

print('end of script.')
