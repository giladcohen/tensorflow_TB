import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

norm_vec       = ['L2', 'L1']
percentage_vec = ['0.5', '10', '100']
input_vec      = ['image', 'embedding']
n_vec          = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for NORM in norm_vec:
    for PERCENTAGE in percentage_vec:
        for INPUT in input_vec:
            for n in n_vec:
                if NORM == 'L2' and float(PERCENTAGE) < 100:
                    continue
                cmd = 'python plots/knn_bayes/cifar10_cats_v_dogs_w_dropout/calc_lipschits_constant.py' + \
                      ' --NORM ' + NORM + \
                      ' --PERCENTAGE ' + PERCENTAGE + \
                      ' --INPUT ' + INPUT + \
                      ' --DATASET_NAME cifar10_cats_v_dogs' + \
                      ' --n ' + str(n)
                run_cmd(cmd)

print('end of script.')
