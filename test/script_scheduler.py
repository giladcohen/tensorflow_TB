import subprocess
import time

# logs_wrn28-10_1513_220317
# logs_wrn28-10_0137_050417
# logs_wrn28-10_0203_090417
# logs_wrn28-10_0853_120417
# logs_wrn28-10_0732_140417
# logs_wrn28-10_1434_160417
# logs_wrn28-10_1753_170417

logs_vec = ['logs/logs_wrn28-10_1513_220317' , \
            'logs/logs_wrn28-10_0137_050417' , \
            'logs/logs_wrn28-10_0203_090417' , \
            'logs/logs_wrn28-10_0853_120417' , \
            'logs/logs_wrn28-10_0732_140417' , \
            'logs/logs_wrn28-10_1434_160417' , \
            'logs/logs_wrn28-10_1753_170417' ]

for i in range(len(logs_vec)):
    cmd = 'python test/save_only_pool.py' + \
          ' --log_root='+logs_vec[i] + \
          ' --eval_data=test'
    print ('start running command %s ' %(cmd))
    process = subprocess.call(cmd, shell=True)
    time.sleep(20)
    cmd = 'python test/save_only_pool.py' + \
          ' --log_root='+logs_vec[i] + \
          ' --eval_data=train'
    print ('start running command %s ' %(cmd))
    process = subprocess.call(cmd, shell=True)
    time.sleep(20)