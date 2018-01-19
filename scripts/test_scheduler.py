import subprocess
import time

logdir_vec   = ['/data/gilad/logs/log_0532_011217_simple-SUPERSEED=01121700',
                '/data/gilad/logs/log_0532_011217_simple-SUPERSEED=01121701',
                '/data/gilad/logs/log_0532_011217_simple-SUPERSEED=01121702',
                '/data/gilad/logs/log_0532_011217_simple-SUPERSEED=01121703',
                '/data/gilad/logs/log_0532_011217_simple-SUPERSEED=01121704',
                '/data/gilad/logs/log_0336_061217_simple-SUPERSEED=06121700',
                '/data/gilad/logs/log_0336_061217_simple-SUPERSEED=06121701',
                '/data/gilad/logs/log_0336_061217_simple-SUPERSEED=06121702',
                '/data/gilad/logs/log_0336_061217_simple-SUPERSEED=06121703',
                '/data/gilad/logs/log_0336_061217_simple-SUPERSEED=06121704',

                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121700',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121701',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121702',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121703',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121704',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121705',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121706',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121707',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121708',
                '/data/gilad/logs/log_0439_171217_cifar100_simple-SUPERSEED=17121709',

                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121700',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121701',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121702',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121703',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121704',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121705',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121706',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121707',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121708',
                '/data/gilad/logs/log_2025_201217_cifar100_simple_dropout_0.5_wd_0.0003-SUPERSEED=20121709',
                ]

weights_vec  = ['uniform', 'distance']
norm_vec     = ['L2', 'L1']
PCA_dims_vec = ['640', '64']
K_vec        = ['1', '5', '30', '150', '1000']

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

for logdir in logdir_vec:
    # create dump once:
    cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/test_automated.py' + \
          ' --ROOT_DIR ' + logdir + \
          ' --KNN_WEIGHTS ' + weights_vec[0] + \
          ' --KNN_NORM ' + norm_vec[0] + \
          ' --PCA_REDUCTION True' + \
          ' --PCA_EMBEDDING_DIMS 3' + \
          ' --KNN_NEIGHBORS 31' + \
          ' --DUMP_NET True' + \
          ' --LOAD_FROM_DISK False' + \
          ' -c examples/test_simple.ini'
    run_cmd(cmd)

    for weights in weights_vec:
        for norm in norm_vec:
            for PCA_dims in PCA_dims_vec:
                for K in K_vec:
                    PCA_REDUCTION = 'False' if int(PCA_dims) == 640 else 'True'
                    cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/test_automated.py' + \
                          ' --ROOT_DIR ' + logdir + \
                          ' --KNN_WEIGHTS ' + weights + \
                          ' --KNN_NORM ' + norm + \
                          ' --PCA_REDUCTION ' + PCA_REDUCTION +\
                          ' --PCA_EMBEDDING_DIMS ' + PCA_dims +\
                          ' --KNN_NEIGHBORS ' + K +\
                          ' --DUMP_NET False' + \
                          ' --LOAD_FROM_DISK True' + \
                          ' -c examples/test_simple.ini'
                    run_cmd(cmd)

print('end of script.')
