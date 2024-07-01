from cgi import test
from tokenize import Double


from itertools import combinations

import matplotlib.pyplot as plt


import numpy as np




from src import HMC_Single_Star
if __name__=="__main__":
    hmc=HMC_Single_Star.HMC_Sampler()
    #hmc.run_svi()
    #hmc.plot_profile()
    import pandas as pd





    err_file='/Users/mattocallaghan/XPNorm/Data/err_black_ps_1'
    err=pd.read_csv(err_file)[['mu_error','ra_error','dec_error']].values

    error=[]
    ebvs=[]
    for i in range(len(hmc.data)):
        ebv,err=hmc.run_model_single(i)
        print(ebv)
        print(err)
        ebvs.append((ebv))
        error.append(err)
    np.save('ebv_black',np.array(ebvs))
    np.save('error_red',np.array(error))
    mean=np.load('ebv_black.npy')
    stds=np.load('error_black.npy')

'''

    plt.figure(figsize=(12, 6))

    # Plot on the left
    plt.subplot(1, 2, 1)
    plt.errorbar(hmc.distance, mean / 3.1, yerr=stds / 3.1, fmt='o', markersize=3, capsize=3, label='Data with Errors')
    plt.scatter(hmc.distance, mean / 3.1, c='r', label='Mean Data')
    plt.xlabel('Distance pc', fontsize=15)
    plt.ylabel('$E(B-V)$', fontsize=15)
    plt.grid(True)
    plt.title('One Cloud', fontsize=15)
    mu=pd.read_csv('/Users/mattocallaghan/XPNorm/Data/data_black_ps_1')[['mu']].values
    mean=np.load('ebv_black.npy')
    stds=np.load('error_black.npy')
    # Plot on the right (same as left)
    plt.subplot(1, 2, 2)
    plt.errorbar(10**((mu+5)/5), mean / 3.1, yerr=stds / 3.1, fmt='o', markersize=3, capsize=3)
    plt.scatter(10**((mu+5)/5), mean / 3.1, c='r')
    plt.xlabel('Distance pc', fontsize=15)
    plt.ylabel('$E(B-V)$', fontsize=15)
    plt.title('Two Clouds', fontsize=15)
    plt.grid(True)

    plt.tight_layout()

    plt.show()

    plt.hist(stds)
    plt.show()
'''

mu=pd.read_csv('/Users/mattocallaghan/XPNorm/Data/data_black_ps_1')[['mu']].values
ra=pd.read_csv('/Users/mattocallaghan/XPNorm/Data/data_black_ps_1')[['dec']].values
mean=np.load('ebv_red.npy')
stds=np.load('error_red.npy')

plt.figure()
#plt.errorbar(hmc.distance, mean / 3.1, yerr=stds / 3.1, fmt='o', markersize=3, capsize=3, label='Data with Errors')
plt.scatter(hmc.distance, mean / 3.1, c='r', label='Mean Data')
plt.xlabel('Distance pc', fontsize=15)
plt.ylabel('$E(B-V)$', fontsize=15)
plt.grid(True)
plt.title('One Cloud', fontsize=15)
plt.show()