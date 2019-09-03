import subprocess
import time
import os

def run_cmd(cmd):
    print ('start running command {}'.format(cmd))
    process = subprocess.call(cmd, shell=True)
    print ('finished running command {}'.format(cmd))
    time.sleep(3)

# for FGSM:
# cmd = 'CUDA_VISIBLE_DEVICES=0 python tensorflow_TB/scripts/extract_characteristics.py' + \
#       ' --dataset cifar10 --attack jsma --targeted True --characteristics nnif --max_indices 50'
# run_cmd(cmd)
#
# cmd = 'CUDA_VISIBLE_DEVICES=0 python tensorflow_TB/scripts/extract_characteristics.py' + \
#       ' --dataset cifar10 --attack deepfool --characteristics nnif --max_indices 50'
# run_cmd(cmd)
#
# cmd = 'CUDA_VISIBLE_DEVICES=0 python tensorflow_TB/scripts/extract_characteristics.py' + \
#       ' --dataset cifar10 --attack cw --targeted True --characteristics nnif --max_indices 50'
# run_cmd(cmd)
#
# cmd = 'CUDA_VISIBLE_DEVICES=0 python tensorflow_TB/scripts/extract_characteristics.py' + \
#       ' --dataset cifar100 --attack deepfool --characteristics nnif --max_indices 30'
# run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset cifar100 --attack cw --targeted True --characteristics nnif --max_indices 30'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset svhn --attack jsma --targeted True --characteristics nnif --max_indices 300'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset svhn --attack cw --targeted True --characteristics nnif --max_indices 300'
run_cmd(cmd)



# for JSMA:
cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset cifar10 --attack fgsm --targeted True --characteristics nnif --max_indices 200'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset cifar10 --attack deepfool --characteristics nnif --max_indices 200'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset svhn --attack fgsm --targeted True --characteristics nnif --max_indices 50'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset svhn --attack deepfool --characteristics nnif --max_indices 50'
run_cmd(cmd)


# for Deepfool:
cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset cifar10 --attack fgsm --targeted True --characteristics nnif --max_indices 100'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset cifar10 --attack jsma --targeted True --characteristics nnif --max_indices 100'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset cifar10 --attack cw --targeted True --characteristics nnif --max_indices 100'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset cifar100 --attack fgsm --targeted True --characteristics nnif --max_indices 40'
run_cmd(cmd)

cmd = 'CUDA_VISIBLE_DEVICES=1 python tensorflow_TB/scripts/extract_characteristics.py' + \
      ' --dataset cifar100 --attack jsma --targeted True --characteristics nnif --max_indices 40'
run_cmd(cmd)


# for CW:
# None

print('end of script.')
