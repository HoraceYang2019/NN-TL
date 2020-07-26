# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:20:34 2020

EMD: Earth Mover's Distance
https://towardsdatascience.com/earth-movers-distance-68fff0363ef2

KLD: Kullback-Leibler Divergence

@author: ASUS
"""
from scipy.stats import wasserstein_distance, entropy
import numpy as np
from scipy.stats import binned_statistic
import random
import matplotlib.pyplot as plt

# In[1]: 1D distribution
np.random.seed(1)

def data_gen(mode='normal', data_n=100, par1=0, par2=1, fixed_bin=False, bin_n=10):
    tiny = 0.00000001    
    intervals  = np.linspace(par1 - 2*par2, par1 + 2*par2, bin_n) 
    
    # what a data distribution will to be generated?
    if mode == 'normal':
        data = [np.random.normal(par1, par2) for _ in range(data_n)]
    if mode == 'lognormal':
        data = [random.lognormvariate(par1, par2) for _ in range(data_n)]
    if mode =='beta':
        data = [random.betavariate(par1, par2) for _ in range(data_n)]
        
    # scipy.stats.binned_statistic(x, values, statistic='mean', bins=10, range=None)
    if fixed_bin == False:
        # using dynamic bin interval
        bins = binned_statistic(data, data,'count', bins = intervals)
    else:
        # using a fixed bin number
        bins = binned_statistic(data, data,'count', bins = bin_n)
    dist = bins[0]/data_n # let sum of dist = 1 
    dist[dist < tiny] = tiny # avoid nan
    return data, dist
# In[2]
bin_number = 15
plt.subplot(1,2,1)
mu = 1; sigma =0.6
x, px = data_gen(mode='normal', data_n=100, par1=mu, par2=sigma, fixed_bin=False, bin_n=bin_number)
plt.hist(x,bin_number)
#print (px)

# In[3]
plt.subplot(1,2,2)
mu = 0.5; sigma =0.8
y, py = data_gen(mode='beta', data_n=250, par1=mu, par2=sigma, fixed_bin=False, bin_n=bin_number)
plt.hist(y,bin_number)
plt.show()

# KL = 0.0
# for i in range(10):
#     KL += px[i] * np.log(px[i] / py[i])
#     # print(str(px[i]) + ' ' + str(py[i]) + ' ' + str(px[i] * np.log(px[i] / py[i])))
# print(KL)
print('KLD:', entropy(px, py))

# scipy.stats.wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None)
# d = wasserstein_distance([3.4, 3.9, 7.5, 7.8], [4.5, 1.4],[1.4, 0.9, 3.1, 7.2], [3.2, 3.5])
#data 1: [3.4, 3.9, 7.5, 7.8] with weigthts = [1.4, 0.9, 3.1, 7.2]；
#data 2: [4.5, 1.4] with weights = [3.2, 3.5]
print('equal weights of EMD:', wasserstein_distance(px, py)) 

weights = [10,5,2,1,1,1,2,5,10]
print('various weights of EMD:', wasserstein_distance(px,weights, py,weights))