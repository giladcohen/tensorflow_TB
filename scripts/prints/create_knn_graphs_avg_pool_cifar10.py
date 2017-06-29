import matplotlib.pyplot as plt
import numpy as np
import os.path
import pylab # use: pylab.savefig(os.path.join(BASE_PATH,'pool_out_dist.png'))
print('start creating graphs')

BASE_PATH = '/home/gilad/workspace/Resnet_KNN'
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000, 3000, 4000, 5000]
NS = 14 #number of samples
NETWORK = "GAP CIFAR10"


'''ENSEMBLE'''
'''printing fc1_vote'''
fc1_vote = [96.03, 96.05, 96.05, 96.07, 96.06, 96.06, 96.04, \
            96.03, 96.05, 96.04, 96.03, 96.05, 96.03, 96.05, \
            96.06, 96.06, 96.07 ,96.07, 96.07]
plt.plot(k[0:NS], fc1_vote[0:NS], 'bo')
plt.plot(k[0:NS], 95.98*np.ones(NS),'r', lw=3)
plt.title('fc1_vote for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist raw'''
fc1_dist_raw = [95.81, 95.81, 95.81, 95.81, 95.81, 95.81, 95.86, \
                95.80, 95.84, 95.82, 95.89, 95.90, 95.87, 95.91, \
                95.88, 95.87, 95.86, 95.84, 95.82]
plt.plot(k[0:NS], fc1_dist_raw[0:NS], 'bo')
plt.plot(k[0:NS], 95.98*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(raw) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist typ'''
fc1_dist_typ = [95.80, 95.80, 95.80, 95.80, 95.80, 95.80, 95.81, \
                95.81, 95.88, 95.91, 95.87, 95.90, 95.89, 95.92, \
                95.86, 95.84, 95.84, 95.85, 95.84]
plt.plot(k[0:NS], fc1_dist_typ[0:NS], 'bo')
plt.plot(k[0:NS], 95.98*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(typ) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist norm1'''
fc1_dist_norm1 = [94.97, 95.63, 95.94, 96.04, 95.79, 95.95, 96.04, \
                  96.05, 95.92, 95.82, 95.86, 95.87, 95.96, 95.87, \
                  95.85, 95.80, 95.84, 95.80, 95.79]
plt.plot(k[0:NS], fc1_dist_norm1[0:NS], 'bo')
plt.plot(k[0:NS], 95.98*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(norm1) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_dist norm2'''
fc1_dist_norm2 = [95.84, 95.84, 95.84, 95.84, 95.84, 95.84, 95.86,
                  95.91, 95.92, 95.90, 95.93, 95.95, 95.91, 95.89,
                  95.95, 95.98, 95.92, 95.94, 95.94]
plt.plot(k[0:NS], fc1_dist_norm2[0:NS], 'bo')
plt.plot(k[0:NS], 95.98*np.ones(NS),'r', lw=3)
plt.title('fc1_dist(norm2) for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing fc1_cons'''
fc1_cons = [96.03, 96.06, 95.98, 95.90, 95.90, 95.93, 95.97, \
                  96.00, 95.97, 95.90, 95.96, 95.96, 96.00, 95.98, \
                  95.99, 95.94, 95.97, 95.92, 95.97]
plt.plot(k[0:NS], fc1_cons[0:NS], 'bo')
plt.plot(k[0:NS], 95.98*np.ones(NS),'r', lw=3)
plt.title('fc1_cons for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing democracy'''
democracy = [96.00, 95.98, 95.97, 95.97, 96.01, 95.97, 95.93, \
             95.96, 95.99, 95.94, 95.97, 95.95, 95.93, 95.91, \
             95.87, 95.86, 95.87, 95.86, 95.87]
plt.plot(k[0:NS], democracy[0:NS], 'bo')
plt.plot(k[0:NS], 95.98*np.ones(NS),'r', lw=3)
plt.title('democracy for %s' %NETWORK)
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

'''printing global_democracy'''
global_democracy = [96.03, 96.03, 96.01, 96.02, 96.01, 96.02, 96.02, \
             96.02, 96.01, 96.00, 96.01, 96.02, 96.02, 96.02, \
             96.03, 96.03, 96.03, 96.03, 96.03]
plt.plot(k[0:NS], global_democracy[0:NS], 'bo')
plt.plot(k[0:NS], 95.98*np.ones(NS),'r', lw=3)
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
plt.plot(k[0:NS], 95.98*np.ones(NS),      'r',    label='neural network ensemble', lw=3)
plt.axis([-24, 525, 95.77, 96.1])
plt.legend()


'''individual networks'''
'''logs_wrn28-10_0853_120417 - vote'''
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000,3000,4000,5000]
pool_out_vote = [95.24, 95.42, 95.45, 95.47, 95.40, 95.42, 95.41, 95.44, \
                 95.39, 95.40, 95.43, 95.42, 95.40, 95.41, 95.42, 95.41, \
                 95.39, 95.39, 95.39]
plt.plot(k, pool_out_vote, 'bo')
plt.plot(k, 95.41*np.ones(len(k)),'r', lw=3)
plt.axis([-50, 5050, 95.22, 95.49])
plt.title('pool_out_vote - (0853_120417) individual.')
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()

''' logs_wrn28-10_0853_120417 - democracy'''
k = [1,3,5,7,9,11,31,51,101,151,201,301,401,501,1000,2000,3000,4000,5000]
pool_out_democracy = [95.44, 95.44, 95.45, 95.49, 95.40, 95.41, 95.43, 95.44, \
                      95.40, 95.42, 95.43, 95.43, 95.40, 95.41, 95.42, 95.41, \
                      95.39, 95.39, 95.39]
plt.plot(k, pool_out_democracy, 'bo')
plt.plot(k, 95.41*np.ones(len(k)),'r', lw=3)
plt.axis([-50, 5050, 95.22, 95.49])
plt.title('pool_out_democracy - (0853_120417) individual.')
plt.xlabel('k')
plt.ylabel('accuracy [%]')
plt.grid(True)
plt.close()
