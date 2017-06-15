import matplotlib.pyplot as plt
import numpy as np
import os.path
import pylab # use: pylab.savefig(os.path.join(BASE_PATH,'pool_out_dist.png'))
print('start creating graphs')

BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000, 3000, 4000, 5000]
NS = 14 #number of samples
NETWORK = "MP CIFAR10"

'''ENSEMBLE'''
'''printing fc1_vote'''
fc1_vote = [95.73, 95.83, 95.81, 95.83, 95.86, 95.84, 95.80, \
            95.83, 95.82, 95.78, 95.80, 95.78, 95.77, 95.76, \
            95.72, 95.67, 95.67, 95.65, 95.62]
plt.plot(k[0:NS], fc1_vote[0:NS], 'bo')
plt.plot(k[0:NS], 95.72*np.ones(NS),'r', lw=3)
plt.title('fc1_vote for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist raw'''
fc1_dist_raw = [95.70, 95.70, 95.70, 95.70, 95.70, 95.70, 95.62, \
                95.62, 95.67, 95.64, 95.62, 95.60, 95.67, 95.69, \
                95.70, 95.71, 95.73, 95.74, 95.74]
plt.plot(k[0:NS], fc1_dist_raw[0:NS], 'bo')
plt.plot(k[0:NS], 95.72*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(raw) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist typ'''
fc1_dist_typ = [95.62, 95.62, 95.62, 95.62, 95.62, 95.62, 95.61, \
                95.63, 95.66, 95.67, 95.68, 95.72, 95.69, 95.72, \
                95.72, 95.80, 95.76, 95.74, 95.78]
plt.plot(k[0:NS], fc1_dist_typ[0:NS], 'bo')
plt.plot(k[0:NS], 95.72*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(typ) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist norm1'''
fc1_dist_norm1 = [95.15, 95.60, 95.77, 95.69, 95.59, 95.65, 95.67, \
                  95.73, 95.59, 95.68, 95.69, 95.66, 95.66, 95.69, \
                  95.67, 95.57, 95.55, 95.53, 95.57]
plt.plot(k[0:NS], fc1_dist_norm1[0:NS], 'bo')
plt.plot(k[0:NS], 95.72*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(norm1) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist norm2'''
fc1_dist_norm2 = [95.69, 95.69, 95.69, 95.69, 95.69, 95.69, 95.74, \
                  95.76, 95.72, 95.74, 95.69, 95.67, 95.69, 95.70, \
                  95.72, 95.71, 95.73, 95.72, 95.73]
plt.plot(k[0:NS], fc1_dist_norm2[0:NS], 'bo')
plt.plot(k[0:NS], 95.72*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(norm2) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_cons'''
fc1_cons = [95.73, 95.79, 95.83, 95.82, 95.79, 95.81, 95.74, \
            95.74, 95.75, 95.76, 95.78, 95.77, 95.75, 95.73, \
            95.71, 95.74, 95.67, 95.62, 95.60]
plt.plot(k[0:NS], fc1_cons[0:NS], 'bo')
plt.plot(k[0:NS], 95.72*np.ones(NS),'r', lw=3)
plt.title('fc1_cons for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing democracy'''
democracy = [95.77, 95.81, 95.80, 95.78, 95.78, 95.79, 95.70, \
             95.72, 95.75, 95.72, 95.75, 95.74, 95.77, 95.82, \
             95.80, 95.74, 95.71, 95.71, 95.68]
plt.plot(k[0:NS], democracy[0:NS], 'bo')
plt.plot(k[0:NS], 95.72*np.ones(NS),'r', lw=3)
plt.title('democracy for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing global_democracy'''
global_democracy = [95.81, 95.77, 95.78, 95.78, 95.78, 95.77, 95.77, \
                    95.76, 95.77, 95.77, 95.78, 95.78, 95.78, 95.77, \
                    95.75, 95.74, 95.74, 95.74, 95.75]
plt.plot(k[0:NS], global_democracy[0:NS], 'bo')
plt.plot(k[0:NS], 95.72*np.ones(NS),'r', lw=3)
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
plt.plot(k[0:NS], 95.72*np.ones(NS),      'r',    label='neural network ensemble', lw=3)
plt.axis([-24, 525, 95.58, 95.88])
plt.legend()

#individual networks:
# logs_wrn28-10_1309_010517 - vote
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000,3000,4000,5000]
pool_out_vote = [94.94, 94.95, 95.05, 95.07, 95.02, 95.03, 95.10, 95.06, \
                 95.06, 95.04, 95.03, 95.06, 95.03, 94.99, 95.01, 94.89, \
                 94.91, 94.90, 94.78]
plt.plot(k, pool_out_vote, 'bo')
plt.plot(k, 95.12*np.ones(len(k)),'r', lw=3)
plt.axis([-50, 5050, 94.75, 95.18])
plt.title('pool_out_vote - (1309_010517) individual.')
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
pylab.savefig(os.path.join(BASE_PATH,'MP_CIFAR10_vote_indiv_1309_010517.png'))
plt.close()

# logs_wrn28-10_1309_010517 - democracy
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000,3000,4000,5000]
pool_out_democracy = [94.96, 94.99, 95.10, 95.10, 95.07, 95.08, 95.12, 95.10, \
                      95.08, 95.02, 95.05, 95.07, 95.04, 95.00, 95.02, 94.91, \
                      94.93, 94.92, 94.81]
plt.plot(k, pool_out_democracy, 'bo')
plt.plot(k, 95.12*np.ones(len(k)),'r', lw=3)
plt.axis([-50, 5050, 94.75, 95.18])
plt.title('pool_out_democracy - (1309_010517) individual.')
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
pylab.savefig(os.path.join(BASE_PATH,'MP_CIFAR10_democracy_indiv_1309_010517.png'))
plt.close()



