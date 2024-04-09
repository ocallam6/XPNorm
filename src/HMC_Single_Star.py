
from turtle import forward
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import jax
import pandas as pd
import os
import matplotlib.pyplot as plt

from numpyro.infer import SVI, Trace_ELBO
import numpyro.optim as optim
from numpyro.infer import (
MCMC, NUTS, SVI, autoguide,
Trace_ELBO, Predictive, autoguide)
from torch import nn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from src.jax_to_numpyro import distribution_to_numpyro
from src.NF_Jax import JaxNormFlow
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import numpyro
from jax import random

from numpyro.infer import MCMC, NUTS, SA, HMC
from numpyro import handlers
import equinox as eqx

'''
HMC 

'''

#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data_full'
test_file='/Users/mattocallaghan/XPNorm/Data/data_noext_1'
test_file='/Users/mattocallaghan/XPNorm/Data/data_black_1'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle'

#test_file='/Users/mattocallaghan/XPNorm/Data/data_red_circle'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle_20'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_noext'

err_file='/Users/mattocallaghan/XPNorm/Data/err_black_1'




class HMC_Sampler():
    def __init__(self,csv_location=data_file,resample=32,*args, **kwargs):


        '''
Learning the extinction map

        '''
#############################################################################
###################### DATA IMPORT #######################################
#############################################################################

        self.normalising_flow=JaxNormFlow()

        self.dist_nf=distribution_to_numpyro(self.normalising_flow.flow)

        #self.data=pd.read_csv(csv_location)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']].values

        self.data_test=pd.read_csv(test_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']].values

        self.data_err=pd.read_csv(err_file)[['mu_error','g_error','bp_error','rp_error','j_msigcom','h_msigcom','ks_msigcom']].values


        self.data_transform=jnp.array([
                                [1., 0., 0., 0., 0., 0.,0],
                                [-1., 1., 0., 0., 0., 0.,0],
                                [0., 0., 1., 0., 0., 0.,-1],
                                [0., 0., 0., 0., 1., 0.,-1],
                                [0., 0., 0., 1., 0., 0.,-1],
                                [0., 0., 0., 0., 1., 0.,-1],
                                [0., 1., 0., 0., 0., 0.,-1]])
        

        self.data=jnp.einsum('ij,bj->bi',jnp.array(self.data_transform),jnp.array(self.data_test))
        self.error=jnp.stack([jnp.diag(arr**2) for arr in self.data_err])

        self.error=(jnp.einsum('ik,bkj->bij',(jnp.array(self.data_transform)),self.error))

        self.error=(jnp.einsum('bik,kj->bij',self.error,jnp.array(self.data_transform).transpose()))
        #self.data=self.data[(self.data[:,1]<10)*(self.data[:,1]>-2)]
        #self.data=self.data[(self.data[:,0]<20)]#*(self.data[:,0]>2)]
        self.distance=10**((self.data[:,0]+5)/5)
        self.data=self.data[:,1:]
        self.mean=jnp.mean(self.data,axis=0)
        self.std=jnp.std(self.data,axis=0)
        self.data=(self.data-self.normalising_flow.mean)

        self.error=self.error[:,1:,1:]


#############################################################################
###################### Extinction Law #######################################
#############################################################################

        self.coeffs= jnp.array([
        [0.194059063720102, -0.000337880688254366, 0.000405004510990789, -0.000119030825664077, -2.90629429374213e-05, 9.85596051245887e-09, 1.22296149043372e-10, 0, 0, 0],
        [0.255058871064972, 7.19016588950228e-05, -0.000250455702483274, 3.99422163967702e-05, -6.83632867675118e-05, -2.3163568526324e-09, 7.26631781961228e-10, -2.27788077229475e-07, 4.97609167483581e-07, 6.67076034469308e-09],
        [0.340301468237771, -0.000826269158576803, -0.000283696380615497, 0.000103538996307887, -0.000156039957793959, 1.81587525109325e-07, 2.33928990111011e-09, 1.63733498839871e-06, 5.71693287820809e-08, 2.954302715354e-08],
        [0.663374149569189, -0.0184814428792349, 0.0100266536020118, -0.00317119320308867, -0.00649336268398495, 3.27674122186053e-05, 1.57894227641494e-06, -0.000116290469708794, 0.000117366662432525, 6.91273258513903e-06],
        [1.1516008149802, -0.0502982507379423, -0.00801054248601918, 0.0028487377407222, -0.0222045923218139, 0.000841943191161668, -1.31018008013547e-05, 0.00346423295251231, -0.000145621334026214, -6.85718568409361e-05],
        [0.993548688906439, -0.110149052160837, 0.0264447715065468, -0.00571010222810317, -0.0374363031107716, 0.00151447309438712, -2.52364537395156e-05, 0.00623824755961677, -0.000123598316318183, -0.000158499801004388]
        ])




    def extinction_coeff(self,x,a,i):

        coeff=self.coeffs[i]
        result=coeff[0]*jnp.ones_like(x)
        result+=coeff[1]*x
        result+=coeff[2]*x**2
        result+=coeff[3]*x**3

        result+=coeff[4]*a
        result+=coeff[5]*a**2
        result+=coeff[6]*a**3

        result+=coeff[7]*x*a
        result+=coeff[8]*a*x**2
        result+=coeff[9]*x*a**2
        return result





#############################################################################
###################### Sampler#######################################
#############################################################################


    def model(self,data,err):
        # Prior distribution for the mean

        with numpyro.plate('data', len(data)):
            a0 = numpyro.sample('a0', dist.Uniform(-0.1,0.8))
            x = numpyro.sample('x', self.dist_nf)
            extinction_vector=jnp.stack([self.extinction_coeff((x+self.normalising_flow.mean)[:,-1],a0,i) for i in jnp.arange(5,-1,-1)]).T
            extinction_vector=(jnp.einsum('ij,bj->bi',(self.normalising_flow.data_transform[1:,1:]),extinction_vector))
            
            x_obs=x+(a0[:,None]*extinction_vector)
 
            numpyro.sample('obs', dist.MultivariateNormal(loc=x_obs[:,:],covariance_matrix=err+1e-5*jnp.eye(6)), obs=data)
            # Observed data is sampled from a Gaussian distribution

    def run_model(self):

        nuts_kernel = NUTS(self.model)

        mcmc = MCMC(nuts_kernel, num_warmup=2000, num_samples=2000,num_chains=jax.local_device_count())


        rng_key = random.PRNGKey(0)

        mcmc.run(rng_key, self.data,self.error)
        
        np.save('/Users/mattocallaghan/XPNorm/Data/a0_black',np.array(mcmc.get_samples()['a0']))

    def run_svi(self):
        guide = autoguide.AutoNormal(self.model)
        rng_key = random.PRNGKey(0)
        svi = SVI(
            model=self.model, 
            data=self.data[0:1],
            err=self.error[0:1],
            guide=guide, 
            optim=optim.Adam(step_size=0.001), 
            loss=Trace_ELBO()
        )
        svi_result = svi.run(
            rng_key=rng_key, 
            num_steps=100000
        )
        params = svi_result.params
        posteriors = guide.sample_posterior(rng_key, params, sample_shape=(10000,))
        np.save('/Users/mattocallaghan/XPNorm/Data/a0_black',np.array(posteriors['a0']))

    def plot_profile(self):
        samples=np.load('/Users/mattocallaghan/XPNorm/Data/a0_black.npy')
        print(samples.shape)
        mean=samples.mean(0)
        stds=samples.std(0)
        print(mean)
        print(stds)
        plt.errorbar(self.distance, mean/3.1, yerr=stds/3.1, fmt='o', markersize=3, capsize=3)
        plt.scatter(self.distance, mean/3.1,c='r')

        plt.xlabel('Distance Mean')
        plt.xlim(0,2000)
        #plt.ylim(0,0.3)
        plt.ylabel('EBV Mean')
        plt.title('Scatter plot with error bars')
        plt.grid(True)
        plt.show()