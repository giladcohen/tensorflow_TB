import subprocess
import time

def run_cmd(cmd):
    process = subprocess.call(cmd, shell=True)

user = 'sivan'
mask = '0xf'
cmd = 'for p in $(pgrep -u {}):; do taskset -p {} $p; done'.format(user, mask)

while True:
    run_cmd(cmd)
    time.sleep(60)

