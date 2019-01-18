import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

norm_vec       = ['L2', 'L1']
percentage_vec = ['0.5']
input_vec      = ['embedding']
n_vec          = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

for NORM in norm_vec:
    for PERCENTAGE in percentage_vec:
        for INPUT in input_vec:
            for n in n_vec:
                cmd = 'python plots/knn_bayes/mnist_1v7_w_dropout/calc_lipschits_constant.py' + \
                      ' --NORM ' + NORM + \
                      ' --PERCENTAGE ' + PERCENTAGE + \
                      ' --INPUT ' + INPUT + \
                      ' --n ' + str(n)
                run_cmd(cmd)

print('end of script.')
