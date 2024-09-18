'''
We look to perform inference on the extinction parameter using the assumption that
far enough away from the galactic centre the distribution of stellar parameters is a 
nice function of the cos(d).
https://www.cosmos.esa.int/web/gaia/iow_20200320
We use the nested sampling algorithms to attempt to avoid the jagged nature of any of the functions.
'''
from tabnanny import verbose
from src.NF_Cos_dist import JaxNormFlow


import scipy
import torch
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
data_file='/Users/mattocallaghan/XPNorm/Data/data_full_ps_2'

test_file='/Users/mattocallaghan/XPNorm/Data/data_high_g'
test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle'


err_file='/Users/mattocallaghan/XPNorm/Data/err_high_g'
err_file='/Users/mattocallaghan/XPNorm/Data/err__black_circle'

############################################################################# 
###################### HMC SAMPLER #######################################
#############################################################################



class NS_Sampler():
    def __init__(self,csv_location=data_file,resample=32,*args, **kwargs):


        '''
        Learning the extinction map.
        This constitutes of:
        1) the pre trained normalising flow
        2) 'zero' extinction data
        3) neural network extinction law
        4) matrix to define the color-magnitude law

        '''
        self.twomass=True
        # This converts the jax pre trained normalising flow to a PDF
        self.normalising_flow=JaxNormFlow()
        self.dist_nf=distribution_to_numpyro(self.normalising_flow.flow)

        #############################################################################
        ###################### DATA IMPORT #######################################
        #############################################################################

        self.data=pd.read_csv(test_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag', 'z_mean_psf_mag', 'y_mean_psf_mag','ks_m']]
        self.data_err=pd.read_csv(err_file)[['mu_error','g_error','bp_error','rp_error','j_msigcom','h_msigcom','g_mean_psf_mag_error', 'r_mean_psf_mag_error', 'i_mean_psf_mag_error', 'z_mean_psf_mag_error', 'y_mean_psf_mag_error','ks_msigcom']].values
        self.mu_error=pd.read_csv(err_file)['mu_error']

        #############################################################################
        ###################### Distance Cuts #######################################
        #############################################################################

        self.data=self.data[self.mu_error/self.data['mu']<0.01]
        self.data=self.data[10**((self.data['mu']+5)/5)<4000]


        #############################################################################
        ###################### Mag Cut #######################################
        #############################################################################
        self.data=self.data[self.data['phot_g_mean_mag']>13]
        self.data=self.data[self.data['ks_m']-self.data['mu']<10.0]
        self.data=self.data[self.data['phot_bp_mean_mag']-self.data['phot_rp_mean_mag']>0.0]
        self.mu_error=self.mu_error.loc[self.data.index]


        #############################################################################
        ###################### Galactic Coordinates Retrieval #######################
        #############################################################################

        self.data_err=self.data_err[self.data.index]
        self.coords=pd.read_csv(test_file)[['ra','dec']].loc[self.data.index]
        self.data=self.data.values
        # Get the galactic coordinates of the source
        galactic_coord = SkyCoord(ra=self.coords['ra'].values*u.degree, dec=self.coords['dec'].values*u.degree, frame='icrs')
        galactic_coord = galactic_coord.transform_to('galactic')
        self.l=galactic_coord.l.value
        self.b=galactic_coord.b.value


        #############################################################################
        ###################### DATA TRANSFORM #######################################
        #############################################################################

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
        self.data=jnp.einsum('ij,bj->bi',jnp.array(self.data_transform),jnp.array(self.data))
        # error is already squared!
        self.error=jnp.stack([jnp.diag(arr)**2 for arr in self.data_err])
        self.orig_mu_error=self.error[:,0,0] #this is because of all the nonlinear transformations to the first coordinate
        
        # Transform the error by x^TSx
        self.error=(jnp.einsum('ik,bkj->bij',(jnp.array(self.data_transform)),self.error)) #transforming the errors, we apply the sinb afterwards because the other terms dont depend on it
        self.error=(jnp.einsum('bik,kj->bij',self.error,jnp.array(self.data_transform).transpose()))
        # this doesnt include yet the error for the sin b part.


        #############################################################################
        ###################### Distance Definitions #################################
        #############################################################################
        # Define distance
        self.mu=self.data[:,0]
        self.distance=10**((self.data[:,0]+5)/5)
        self.cos_dist=self.distance*jnp.abs(jnp.sin(jnp.radians(self.b)))
        #tiny bit careful here with the definitions

        #############################################################################
        ######################## FULL DATA PROCESS #################################
        #############################################################################
        if(self.twomass):
            #ERROR
            self.dist_std=10**((self.data[:,0]+5)/5)*(np.log(10)/5)*jnp.sqrt(self.error[:,0,0]) # mu --> distance, stds
            self.dist_error_sin=self.dist_std*(jnp.abs(jnp.sin(jnp.radians(self.b))))  # distance ---> sin_dist, stds
            self.log_dist_error=((5/np.log(10))*(1/self.cos_dist)*self.dist_error_sin)**2 # sin_dist ---> mu, variance
            self.error=self.error.at[:,0,0].set(self.log_dist_error)  

            #DATA
            self.data = self.data.at[:,0].set(2.5*jnp.log10(self.cos_dist**2/100))
            
            
            #self.data=(self.data-self.normalising_flow.mean)
            # actually we will stop taking the mean away and in the inference process add it back


        #############################################################################
        ######################## Gaia + PS Data Process #############################
        #############################################################################
        if(self.twomass==False):
            self.data_transform_2mass=jnp.array([
                                [1., 0., 0., 0., 0., 0.,0,0,0,0,0,0],
                                [-1., 0., 0., 0., 0.,0,0,0,0,0, 1.,0],
                                [0., 1., 0., 0.,0,0,0,0,0, 0., -1.,0],
                                [0., 0., 1., 0., 0,0,0,0,0,0., -1.,0],
                                [0., 0., 0., 1., 0., 0,0,0,0,0,-1.,0],
                                [0., 0., 0., 0., 0., 0,1,0,0,0,-1,0],
                                [0., 0., 0., 0., 0., 0,0,1,0,0,-1,0],
                                [0., 0., 0., 0., 0., 0,0,0,1,0,-1,0],
                                [0., 0., 0., 0., 0., 0,0,0,0,1,-1,0]])
            self.data_transform_inv=jnp.linalg.inv(jnp.array([
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
                                [0., 0., 0., 0., 0., 0,0,0,0,0,1,-1]]))

            # transform the data to be used by without 2mass
            self.data=jnp.einsum('ij,bj->bi',self.data_transform_inv,self.data)
            self.data=jnp.einsum('ij,bj->bi',self.data_transform_2mass,self.data)


            self.error=(jnp.einsum('ik,bkj->bij',(jnp.array(self.data_transform_inv)),self.error))
            self.error=(jnp.einsum('bik,kj->bij',self.error,jnp.array(self.data_transform_inv).transpose()))
            self.error=(jnp.einsum('ik,bkj->bij',(jnp.array(self.data_transform_2mass)),self.error))
            self.error=(jnp.einsum('bik,kj->bij',self.error,jnp.array(self.data_transform_2mass).transpose()))

            # These transformations below are the same as the full data case 
            self.dist_std=10**((self.data[:,0]+5)/5)*(np.log(10)/5)*jnp.sqrt(self.error[:,0,0]) 
            self.dist_error_sin=self.dist_std*(jnp.abs(jnp.sin(jnp.radians(self.b))))
            self.log_dist_error=((5/np.log(10))*(1/self.cos_dist)*self.dist_error_sin)**2 

            self.error=self.error.at[:,0,0].set(self.log_dist_error)  
            self.data = self.data.at[:,0].set(2.5*jnp.log10(self.cos_dist**2/100))
            
            



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
        saved_params = torch.load('/Users/mattocallaghan/XPNorm/Data/model_extinction_2_gauss')
        w1 = jnp.array(saved_params['fc1.weight'].T)  # Transpose weight matrix
        b1 = jnp.array(saved_params['fc1.bias'])
        w2 = jnp.array(saved_params['fc2.weight'].T)  # Transpose weight matrix
        b2 = jnp.array(saved_params['fc2.bias'])
        w3 = jnp.array(saved_params['fc3.weight'].T)  # Transpose weight matrix
        b3 = jnp.array(saved_params['fc3.bias'])
        self.nn_params = (w1, b1, w2, b2,w3,b3)

    def extinction_neural(self, x):
        w1, b1, w2, b2,w3,b3 = self.nn_params
        x = jnp.dot(x, w1) + b1
        x = jnp.maximum(0, x)
        x = jnp.dot(x, w2) + b2
        x = jnp.maximum(0, x)
        x = jnp.dot(x, w3) + b3
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
            z = numpyro.sample('z', dist.Uniform(0.001,1.0).expand([1,12]))
            x=jax.vmap(self.normalising_flow.flow.bijection.transform)(z)+self.normalising_flow.mean

            
            ### Use the extinction neural network to define Ax/A0
            x_observed,x_std=self.extincted_phot(x,a0)
            
            eps=numpyro.sample('eps', dist.MultivariateNormal(0,jnp.eye(12)))
            x_obs = x_observed+eps*x_std

            if(self.twomass==False):
                # take the transfomration and the elements correpsonding to the errors.
                x_obs=x_obs.at[:,1].set(x_obs[:,1]+x_obs[:,-1])
                x_obs=x_obs.at[:,2:-1].set(x_obs[:,2:-1]-x_obs[:,-1])
                x_obs=x_obs[:,[0,1,2,3,4,7,8,9,10]]
                numpyro.sample('obs', dist.MultivariateNormal(loc=x_obs[:,:],covariance_matrix=err+(0.01**2)*jnp.eye(9)), obs=data)
            # Match to data
            else:
                numpyro.sample('obs', dist.MultivariateNormal(loc=x_obs[:,:],covariance_matrix=err+(0.01**2)*jnp.eye(12)), obs=data)
            # Observed data is sampled from a Gaussian distribution
    def extincted_phot(self,x,a0,subtract=False):
            #the input of thiss should be in physical units of magnitudes
            #Neural Network
              
            extinction_input=jnp.concatenate((a0[:,None],(x)[:,jnp.array([2,4])]),1) #picking out gk and rbk
            extinction_input=(extinction_input-jnp.array(self.data_extinction.mean(0)[[0,1,2]]))/jnp.array(self.data_extinction.std(0)[[0,1,2]])
            out=self.extinction_neural(extinction_input) # i got my signs wrong when building the NN
            nn_output,nn_std=out[:,:11],jnp.exp(out[:,11:])
            
            #ext vector

            indices_order=[0,1,2,3,4,6,8,7,10,9,5] #because i didnt match the extinction law correctly
            indices_order_mean_std=[0+3,1+3,2+3,3+3,4+3,6+3,8+3,7+3,10+3,9+3,5+3]
            extinction_vector=jnp.array([0.8,1.1,0.625,0.285,0.18,0.122,1.225,0.66,0.89,0.437,0.5225])[None,jnp.array(indices_order)]
            extinction_vector_std=jnp.array([0.1,0.05,0.025,0.001,0.001,0.001,0.005,0.005,0.005,0.005,0.005])[None,jnp.array(indices_order_mean_std)] 
          

           
           
            extinction_vector=jax.nn.relu(nn_output[:,jnp.array(indices_order)]*jnp.array(self.data_extinction.std(0)[indices_order_mean_std])+jnp.array(self.data_extinction.mean(0)[indices_order_mean_std]))
            extinction_vector=(jnp.einsum('ij,bj->bi',(self.normalising_flow.data_transform[1:,1:]),extinction_vector))
            extinction_vector=jnp.concatenate((jnp.zeros(shape=(extinction_vector.shape[0],1)),extinction_vector),1)

            #[['av',"GK","RPK","k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']].dropna().values
            #'mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag', 'z_mean_psf_mag', 'y_mean_psf_mag','ks_m'



            # the erros here have to be taken seriously!
            extinction_vector_std=jax.nn.relu(nn_std[:,jnp.array(indices_order)]*jnp.array(self.data_extinction.std(0)[indices_order_mean_std]))
            extinction_vector_std=(jnp.einsum('ij,bj->bi',(jnp.abs(self.normalising_flow.data_transform[1:,1:])),extinction_vector_std**2))
            extinction_vector_std=jnp.concatenate((jnp.zeros(shape=(extinction_vector_std.shape[0],1)),jnp.sqrt(extinction_vector_std)),1)
            


            # Theoretical Observed
            x_obs=x+(a0[:,None]*extinction_vector)

            x_std=(a0[:,None]*extinction_vector_std)

            if(subtract):
                x_obs=x-(a0[:,None]*extinction_vector)
            return x_obs,x_std



    def ns_model_single(self,i):
        ns = NestedSampler(self.model)
        ns.run(random.PRNGKey(0),data=self.data[i:i+1],err=self.error[i])
        ns.print_summary()
        # samples obtained from nested sampler are weighted, so
        # we need to provide random key to resample from those weighted samples
        ns_samples = ns.get_samples(random.PRNGKey(1), num_samples=5000)



        return ns_samples
        

    def priors(self,cube):
        # the argument, cube, consists of values from 0 to 1
        # we have to convert them to physical scales

        params = cube.copy()
        # let background level go from -10 to +10
        hi=1.5
        lo=0.00001
        params[0] =  cube[0] * (hi - lo) + lo
        # let amplitude go from 0.1 to 100
        z = scipy.stats.norm.ppf(params[1:][None,:])
        x=np.array(jax.vmap(self.normalising_flow.flow.bijection.transform)(jnp.array(z))+self.normalising_flow.mean)
        params[1:]=x

        return params


    def log_likelihood(self,params):
        data=self.data[0]
        err=self.error[0]
        a0=params[0:1]
        x=params[1:][None,:]

        x_observed,x_std=np.array(self.extincted_phot(jnp.array(x),jnp.array(a0)))
        x_obs = x_observed[0]

        covariance_matrix=np.linalg.inv(err+(0.001**2)*jnp.eye(12))

        l=-0.5*(np.einsum('i,ij->j',(x_obs-data),covariance_matrix)*(x_obs-data)).sum()


        return l



