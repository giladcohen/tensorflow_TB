import sys
import time

orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

for i in range(5):
    print 'i = ', i
    time.sleep(5)  # delays for 5 seconds. You can Also Use Float Value.

sys.stdout = orig_stdout
f.close()