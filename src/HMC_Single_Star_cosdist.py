'''
We look to perform inference on the extinction parameter using the assumption that
far enough away from the galactic centre the distribution of stellar parameters is a 
nice function of the cos(d).
https://www.cosmos.esa.int/web/gaia/iow_20200320
'''
from src.NF_Cos_dist import JaxNormFlow

import torch
from torch import nn
import normflows as nf

import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

import jax
from jax import grad,vmap
from jax import jacfwd, jacrev
import jax.numpy as jnp
from jax import random, hessian


import numpyro.optim as optim
from numpyro.infer import (
MCMC, NUTS, SVI, autoguide,
Trace_ELBO, Predictive, autoguide)
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import numpyro
from numpyro import handlers
from numpyro.contrib.nested_sampling import NestedSampler
import numpyro.distributions as dist

from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from src.jax_to_numpyro import distribution_to_numpyro

from astropy.coordinates import SkyCoord
import astropy.units as u




############################################################################# 
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data_full_ps_1'
test_file='/Users/mattocallaghan/XPNorm/Data/data_noext_ps_1'
test_file='/Users/mattocallaghan/XPNorm/Data/data_bigcirc_ps_1'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_red_1'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle'

#test_file='/Users/mattocallaghan/XPNorm/Data/data_red_circle'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle_20'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_noext'

err_file='/Users/mattocallaghan/XPNorm/Data/err_bigcirc_ps_1'

#err_file='/Users/mattocallaghan/XPNorm/Data/err_noext_ps_1'

############################################################################# 
###################### HMC SAMPLER #######################################
#############################################################################



class HMC_Sampler():
    def __init__(self,csv_location=data_file,resample=32,*args, **kwargs):


        '''
        Learning the extinction map.
        This constitutes of:
        1) the pre trained normalising flow
        2) 'zero' extinction data
        3) neural network extinction law
        4) matrix to define the color-magnitude law

        '''

        # This converts the jax pre trained normalising flow to a PDF
        self.normalising_flow=JaxNormFlow()
        self.dist_nf=distribution_to_numpyro(self.normalising_flow.flow)

#############################################################################
###################### DATA IMPORT #######################################
#############################################################################

        self.data_test=pd.read_csv(test_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag', 'z_mean_psf_mag', 'y_mean_psf_mag','ks_m']]
        self.data_err=pd.read_csv(err_file)[['mu_error','g_error','bp_error','rp_error','j_msigcom','h_msigcom','g_mean_psf_mag_error', 'r_mean_psf_mag_error', 'i_mean_psf_mag_error', 'z_mean_psf_mag_error', 'y_mean_psf_mag_error','ks_msigcom']].values
        self.mu_error=pd.read_csv(err_file)['mu_error']
        self.data_test=self.data_test[self.mu_error/self.data_test['mu']<0.01]
        self.data_test=self.data_test[10**((self.data_test['mu']+5)/5)<4000]
        self.mu_error=self.mu_error.loc[self.data_test.index]
        self.data_test=self.data_test[self.mu_error/self.data_test['mu']<0.01]



        self.data_test=self.data_test[self.data_test['phot_g_mean_mag']>13]

        self.data_test=self.data_test[self.data_test['ks_m']-self.data_test['mu']<10.0]
        self.data_test=self.data_test[self.data_test['phot_bp_mean_mag']-self.data_test['phot_rp_mean_mag']>0.0]

        self.data_err=self.data_err[self.data_test.index]
        self.coords=pd.read_csv(test_file)[['ra','dec']].loc[self.data_test.index]
        self.data_test=self.data_test.values


        # Get the galactic coordinates of the source
        galactic_coord = SkyCoord(ra=self.coords['ra'].values*u.degree, dec=self.coords['dec'].values*u.degree, frame='icrs')
        galactic_coord = galactic_coord.transform_to('galactic')
        self.l=galactic_coord.l.value
        self.b=galactic_coord.b.value


        #This corresponds to an array:
        # [mu,Ks,GK,BPK,RPK,JK,HK,gK,rK,iK,zK,yK]

        self.data_transform=jnp.array([
                                [1., 0., 0., 0., 0., 0.,0,0,0,0,0,0],
                                [-1., 0., 0., 0., 0.,0,0,0,0,0, 0.,1],
                                [0., 1., 0., 0., 0.,0,0,0,0,0, 0.,-1],
                                [0., 0., 1., 0.,0,0,0,0,0, 0., 0.,-1],
                                [0., 0., 0., 1., 0,0,0,0,0,0., 0.,-1],
                                [0., 0., 0., 0., 1., 0,0,0,0,0,0.,-1],
                                [0., 0., 0., 0., 0., 1,0,0,0,0,0,-1],
                                [0., 0., 0., 0., 0., 0,1,0,0,0,0,-1],
                                [0., 0., 0., 0., 0., 0,0,1,0,0,0,-1],
                                [0., 0., 0., 0., 0., 0,0,0,1,0,0,-1],
                                [0., 0., 0., 0., 0., 0,0,0,0,1,0,-1],
                                [0., 0., 0., 0., 0., 0,0,0,0,0,1,-1]])
        
        # Transform the data
        self.data_test=jnp.einsum('ij,bj->bi',jnp.array(self.data_transform),jnp.array(self.data_test))
        # Transform the error by x^TSx
        self.error=jnp.stack([jnp.diag(arr)**2 for arr in self.data_err])
        self.error=self.error.at[:,0,0].set(self.error[:,0,0]*(jnp.abs(jnp.sin(jnp.radians(self.b))**2)))
        self.error=(jnp.einsum('ik,bkj->bij',(jnp.array(self.data_transform)),self.error))
        self.error=(jnp.einsum('bik,kj->bij',self.error,jnp.array(self.data_transform).transpose()))

        # Define distance
        self.distance=10**((self.data_test[:,0]+5)/5)
        self.data = self.data_test.at[:,0].set(self.data_test[:,0]*jnp.abs(jnp.sin(jnp.radians(self.b))))
        self.mean=jnp.mean(self.data_test,axis=0)
        self.std=jnp.std(self.data_test,axis=0)
        self.data=(self.data-self.normalising_flow.mean)


#############################################################################
###################### Extinction Law #######################################
#############################################################################

        data_file='/Users/mattocallaghan/XPNorm/Data/exts.npy'
        data_file_2='/Users/mattocallaghan/XPNorm/Data/vals.npy'

        teff=np.arange(3500,10000,100)
        logg=np.arange(0,5,0.15)
        feh=np.arange(-3,0.5,0.5)
        #Rv=np.arange(1,5,0.2)
        av=np.arange(0.001,5,0.1)

        mesh=np.meshgrid(teff,logg,feh,av)
        teff=mesh[0].flatten()[:,None]
        logg=mesh[1].flatten()[:,None]
        feh=mesh[2].flatten()[:,None]
        av=mesh[3].flatten()[:,None]
        columns = ['teff','feh','logg','av',"a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_WISE_W1','a_WISE_W2','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z',"Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3','2MASS_J','2MASS_H','2MASS_Ks','WISE_W1','WISE_W2','PS_g','PS_i','PS_r','PS_y','PS_z'] #add wise on later.
        self.data_extinction=pd.DataFrame(np.concatenate((teff,logg,feh,av,np.load(data_file),np.load(data_file_2)),1),columns=columns)
        self.data_extinction=self.data_extinction[['av',"Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3','2MASS_J','2MASS_H','2MASS_Ks','PS_g','PS_i','PS_r','PS_y','PS_z',"a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z']].dropna()

        self.data_extinction[["k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']]=(self.data_extinction[["a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z']].values)/self.data_extinction['av'].values[:,None]
        self.data_extinction['GK']=self.data_extinction['Gaia_G_EDR3']-self.data_extinction['2MASS_Ks']
        self.data_extinction['RPK']=self.data_extinction['Gaia_RP_EDR3']-self.data_extinction['2MASS_Ks']
        self.data_extinction=self.data_extinction[['av',"GK","RPK","k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']].dropna().values

        ####
        ### Load the NN
        ####
        saved_params = torch.load('/Users/mattocallaghan/XPNorm/Data/model_extinction')
        w1 = jnp.array(saved_params['fc1.weight'].T)  # Transpose weight matrix
        b1 = jnp.array(saved_params['fc1.bias'])
        w2 = jnp.array(saved_params['fc3.weight'].T)  # Transpose weight matrix
        b2 = jnp.array(saved_params['fc3.bias'])
        self.nn_params = (w1, b1, w2, b2)

    def extinction_neural(self, x):
        w1, b1, w2, b2 = self.nn_params
        x = jnp.dot(x, w1) + b1
        x = jnp.maximum(0, x)
        x = jnp.dot(x, w2) + b2
        return x

    def extinction_coeff(self,x,a,i):
        """
        This does nothing now. But i will leave it here so later can infer the 
        Gaia G band extinction.
        """

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
############################# Sampler########################################
#############################################################################


    def model(self,data,err):
        # Prior distribution for the mean
        with numpyro.plate('data', len(data)):
            a0 = numpyro.sample('a0', dist.Uniform(0.001,1.0))
            x = numpyro.sample('x', self.dist_nf)
            
            ### Use the extinction neural network to define Ax/A0
            extinction_input=jnp.concatenate((a0[:,None],(x+self.normalising_flow.mean)[:,jnp.array([1,3])]),1) #picking out gk and rbk
            extinction_input=(extinction_input-jnp.array(self.data_extinction.mean(0)[[0,1,2]]))/jnp.array(self.data_extinction.std(0)[[0,1,2]])
            nn_output=self.extinction_neural(extinction_input) # i got my signs wrong when building the NN
            indices_order=[0,1,2,3,4,6,8,7,10,9,5] #because i didnt match the extinction law correctly
            indices_order_mean_std=[0+3,1+3,2+3,3+3,4+3,6+3,8+3,7+3,10+3,9+3,5+3]
            extinction_vector=jax.nn.relu(nn_output[:,jnp.array(indices_order)]*jnp.array(self.data_extinction.std(0)[indices_order_mean_std])+jnp.array(self.data_extinction.mean(0)[indices_order_mean_std]))
            extinction_vector=(jnp.einsum('ij,bj->bi',(self.normalising_flow.data_transform[1:,1:]),extinction_vector))
            extinction_vector=jnp.concatenate((jnp.zeros(shape=(extinction_vector.shape[0],1)),extinction_vector),1)
            
            # Theoretical Observed
            x_obs=x+(a0[:,None]*extinction_vector)
            
            # Match to data
            numpyro.sample('obs', dist.MultivariateNormal(loc=x_obs[:,:],covariance_matrix=err+(0.01**2)*jnp.eye(12)), obs=data)
            # Observed data is sampled from a Gaussian distribution



    def run_model(self):

        nuts_kernel = NUTS(self.model)

        mcmc = MCMC(nuts_kernel, num_warmup=2000, num_samples=2000,num_chains=jax.local_device_count())


        rng_key = random.PRNGKey(0)

        mcmc.run(rng_key, self.data,self.error)
        
        np.save('/Users/mattocallaghan/XPNorm/Data/a0_black',np.array(mcmc.get_samples()['a0']))

    def run_model_single(self,i):

        nuts_kernel = NUTS(self.model)

        mcmc = MCMC(nuts_kernel, num_warmup=200, num_samples=200,num_chains=jax.local_device_count())


        rng_key = random.PRNGKey(0)

        mcmc.run(rng_key, self.data[i:i+1],self.error[i])

        return mcmc.get_samples()['a0'][:].mean(),mcmc.get_samples()['a0'][:].std()
        

    def ns_model_single(self,i):
        ns = NestedSampler(self.model)
        ns.run(random.PRNGKey(0),data=self.data[i:i+1],err=self.error[i])
        ns.print_summary()
        # samples obtained from nested sampler are weighted, so
        # we need to provide random key to resample from those weighted samples
        ns_samples = ns.get_samples(random.PRNGKey(1), num_samples=1000)



        return ns_samples['a0'][:].mean(),ns_samples['a0'][:].std()
        



    def run_svi(self):
        guide = autoguide.AutoNormal(self.model)
        rng_key = random.PRNGKey(0)
        svi = SVI(
            model=self.model, 
            data=self.data,
            err=self.error,
            guide=guide, 
            optim=optim.Adam(step_size=0.01), 
            loss=Trace_ELBO()
        )
        svi_result = svi.run(
            rng_key=rng_key, 
            num_steps=10000
        )
        params = svi_result.params
        posteriors = guide.sample_posterior(rng_key, params, sample_shape=(100000,))
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







    def mle(self,i):

        rng_key = random.PRNGKey(0)

        x_original=self.dist_nf.sample(rng_key,(50000,))
        def loss_fn(pars):
            x_orig,a0=pars[:-1][None,:],pars[-1:][None,:]
            extinction_vector=jnp.stack([self.extinction_coeff((x_orig+self.normalising_flow.mean)[:,-1],a0.mean(),i) for i in jnp.arange(5,-1,-1)]).T
            extinction_vector=(jnp.einsum('ij,bj->bi',(self.normalising_flow.data_transform[1:,1:]),extinction_vector))

            x=x_orig+a0*extinction_vector

            
            loss=jnp.einsum('bi,ij->bj',(x-self.data[i:i+1]),jnp.linalg.inv(self.error[i]))
            loss=-0.5*(loss*(x-self.data[i:i+1])).sum(-1)
            return loss[:]
            #loss=loss
            #loss-=jnp.linalg.det(self.error[0])**(0.5)
            #loss-=(2*3.14)**(x.shape[-1]/2)
        grad_loss_fn = vmap(hessian(((loss_fn))))
        all_pars=[]
        for a0 in np.arange(-0.0,1.0,0.01):
            pars=jnp.concatenate((x_original,a0*jnp.ones_like(x_original[:,0])[:,None]),1)
            all_pars.append(pars)     

        pars=jnp.concatenate(all_pars,0)
        loss=vmap(loss_fn)(pars)
        gradient=-1*(grad_loss_fn)(pars)[:,0,:,:]

        idx=np.argmax(loss)
        print(pars[idx,-1])
        print((np.sqrt(np.abs(np.linalg.inv(((gradient)[idx]))[-1,-1]))))
        return pars[idx,-1],np.linalg.inv(((gradient)[idx]))[-1,-1]
        

