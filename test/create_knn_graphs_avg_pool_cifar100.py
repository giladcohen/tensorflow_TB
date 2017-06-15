import matplotlib.pyplot as plt
import numpy as np
import os.path
import pylab # use: pylab.savefig(os.path.join(BASE_PATH,'pool_out_dist.png'))
print('start creating graphs')

BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000, 3000, 4000, 5000]
NS = 14 #number of samples
NETWORK = "GAP CIFAR100"


'''ENSEMBLE'''
'''printing fc1_vote'''
fc1_vote = [81.99, 82.19, 82.12, 82.16, 82.14, 82.11, 82.02, \
            81.99, 81.93, 81.98, 81.90, 81.86, 81.76, 81.67, \
            81.62, 81.54, 80.92, 80.82, 80.75]
plt.plot(k[0:NS], fc1_vote[0:NS], 'bo')
plt.plot(k[0:NS], 82.12*np.ones(NS),'r', lw=3)
plt.title('fc1_vote for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist raw'''
fc1_dist_raw = [81.10, 81.10, 81.10, 81.10, 81.10, 81.10, 81.45, \
                81.72, 81.76, 81.80, 81.90, 81.96, 81.95, 81.81, \
                81.81, 81.75, 81.77, 81.74, 81.78]
plt.plot(k[0:NS], fc1_dist_raw[0:NS], 'bo')
plt.plot(k[0:NS], 82.12*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(raw) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist typ'''
fc1_dist_typ = [81.07, 81.07, 81.07, 81.07, 81.07, 81.07, 81.53, \
                81.66, 81.78, 81.88, 81.85, 81.99, 81.88, 81.84, \
                81.83, 81.78, 87.70, 81.66, 81.75]
plt.plot(k[0:NS], fc1_dist_typ[0:NS], 'bo')
plt.plot(k[0:NS], 82.12*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(typ) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist norm1'''
fc1_dist_norm1 = [79.39, 80.34, 81.25, 81.50, 81.41, 81.49, 81.94, \
                  81.99, 81.88, 81.85, 81.88, 81.81, 81.74, 81.62, \
                  81.36, 81.43, 81.23, 81.15, 81.16]
plt.plot(k[0:NS], fc1_dist_norm1[0:NS], 'bo')
plt.plot(k[0:NS], 82.12*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(norm1) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist norm2'''
fc1_dist_norm2 = [81.48, 81.48, 81.48, 81.48, 81.48, 81.48, 81.90, \
                  81.95, 82.08, 82.00, 82.01, 81.99, 81.92, 81.85, \
                  81.82, 81.71, 81.70, 87.77, 81.74]
plt.plot(k[0:NS], fc1_dist_norm2[0:NS], 'bo')
plt.plot(k[0:NS], 82.12*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(norm2) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_cons'''
fc1_cons = [81.99, 81.90, 81.73, 81.74, 81.64, 81.81, 81.55, \
            81.34, 81.13, 81.06, 80.92, 80.90, 80.71, 80.63, \
            80.22, 79.62, 79.6, 79.39, 79.3]
plt.plot(k[0:NS], fc1_cons[0:NS], 'bo')
plt.plot(k[0:NS], 82.12*np.ones(NS),'r', lw=3)
plt.title('fc1_cons for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing democracy'''
democracy = [81.95, 82.11, 82.07, 81.95, 81.97, 82.02, 81.86, \
             81.93, 81.75, 81.77, 81.71, 81.61, 81.59, 81.48, \
             81.26, 81.15, 81.04, 80.90, 80.89]
plt.plot(k[0:NS], democracy[0:NS], 'bo')
plt.plot(k[0:NS], 82.12*np.ones(NS),'r', lw=3)
plt.title('democracy for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing global_democracy'''
global_democracy = [82.08, 82.28, 82.19, 82.23, 82.26, 82.25, 82.12, \
                    82.11, 82.16, 82.10, 82.09, 82.02, 82.03, 82.03, \
                    81.97, 82.02, 82.10, 82.1, 82.07]
plt.plot(k[0:NS], global_democracy[0:NS], 'bo')
plt.plot(k[0:NS], 82.12*np.ones(NS),'r', lw=3)
plt.title('global_democracy for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

''' comparison'''
plt.plot(k[0:NS], fc1_vote[0:NS],         'g',    label='vote')
plt.plot(k[0:NS], fc1_dist_raw[0:NS],     'k--',  label='dist(raw)')
plt.plot(k[0:NS], fc1_dist_typ[0:NS],     'k',    label='dist(typ)')
plt.plot(k[0:NS], fc1_dist_norm1[0:NS],   'm--',  label='dist(norm1)')
plt.plot(k[0:NS], fc1_dist_norm2[0:NS],   'y',    label='dist(norm2)')
plt.plot(k[0:NS], fc1_cons[0:NS],         'm',    label='cons')
plt.plot(k[0:NS], democracy[0:NS],        'c',    label='democracy')
plt.plot(k[0:NS], global_democracy[0:NS], 'c--',  label='global_democracy')
plt.plot(k[0:NS], 82.12*np.ones(NS),      'r',    label='neural network ensemble', lw=3)
plt.axis([-24, 525, 81.02, 82.34])
plt.legend()

#individual networks:
# logs_wrn28-10_1945_120517 - vote
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000,3000,4000,5000]
pool_out_vote = [78.80, 79.12, 78.98, 78.99, 79.04, 79.00, 79.15, 78.92, \
                 78.89, 78.80, 78.76, 78.65, 78.54, 78.40, 78.25, 77.91, \
                 77.58, 77.51, 77.25]
plt.plot(k, pool_out_vote, 'bo')
plt.plot(k, 79.40*np.ones(len(k)),'r', lw=3)
plt.axis([-50, 5050, 77.22, 79.5])
plt.title('pool_out_vote - (1945_120517) individual.')
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
pylab.savefig(os.path.join(BASE_PATH,'GAP_CIFAR100_vote_indiv_1945_120517.png'))
plt.close()

# logs_wrn28-10_0853_120417 - democracy
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000,3000,4000,5000]
pool_out_democracy = [79.09, 79.25, 79.02, 79.02, 79.09, 79.07, 79.15, 79.01, \
                      78.97, 78.92, 78.83, 78.86, 78.72, 78.57, 78.51, 78.24, \
                      78.13, 78.06, 77.83]
plt.plot(k, pool_out_democracy, 'bo')
plt.plot(k, 79.40*np.ones(len(k)),'r', lw=3)
plt.axis([-50, 5050, 77.22, 79.5])
plt.title('pool_out_democracy - (1945_120517) individual.')
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
pylab.savefig(os.path.join(BASE_PATH,'GAP_CIFAR10_democracy_indiv_0853_120417.png'))
plt.close()
