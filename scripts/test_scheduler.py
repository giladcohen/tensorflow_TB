import subprocess
import time

logdir_vec   = ['/data/gilad/logs/log_1005_051217_simple_dropout_0.5-SUPERSEED=05121700']
weights_vec  = ['uniform', 'distance']
norm_vec     = ['L2', 'L1']
PCA_dims_vec = ['640', '64']
K_vec        = ['1', '5', '30', '150', '1000']

# Just for dumping the files once and calculating the DNN results
cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/test_automated.py' + \
      ' --ROOT_DIR ' + logdir_vec[0] + \
      ' --KNN_WEIGHTS ' + weights_vec[0] + \
      ' --KNN_NORM ' + norm_vec[0] + \
      ' --PCA_REDUCTION True' + \
      ' --PCA_EMBEDDING_DIMS 3' + \
      ' --KNN_NEIGHBORS 10000' + \
      ' --DUMP_NET True' + \
      ' --LOAD_FROM_DISK False' + \
      ' -c examples/test_simple.ini'

print ('start running command {}'.format(cmd))
process = subprocess.call(cmd, shell=True)
print ('finished running command {}'.format(cmd))


# for logdir in logdir_vec:
#     for weights in weights_vec:
#         for norm in norm_vec:
#             for PCA_dims in PCA_dims_vec:
#                 for K in K_vec:
#                     cmd = 'CUDA_VISIBLE_DEVICES=3 python scripts/test.py' + \
#                           ' --train.train_control.ROOT_DIR=' + logdir + \
#                           ' --'
#
#
# for i in range(len(logs_vec)):
#     cmd = 'python test/save_only_pool.py' + \
#           ' --log_root='+logs_vec[i] + \
#           ' --eval_data=test'
#     print ('start running command %s ' %(cmd))
#     process = subprocess.call(cmd, shell=True)
#     time.sleep(20)
#     cmd = 'python test/save_only_pool.py' + \
#           ' --log_root='+logs_vec[i] + \
#           ' --eval_data=train'
#     print ('start running command %s ' %(cmd))
#     process = subprocess.call(cmd, shell=True)
#     time.sleep(20)