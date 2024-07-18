
"""
Here we define a method to learn the cos(d)-color-magnitude diagram using Normalising flows.
We need to make sure that cos(d) is uniform. Well not necesarily but need to be careful with the distance 
Sources closer will be less. 
Need to get rid of outliers like galaxies.
"""

from src.Normalising_Flow_nodist import Normalising_Flow_Trainer

import pandas as pd
import jax.numpy as jnp
import jax.random as jr


from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.tasks import two_moons
from flowjax.train import fit_to_data
import equinox as eqx
from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import numpyro
from astropy.coordinates import SkyCoord
import astropy.units as u

#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data_full_ps_1'
test_file='/Users/mattocallaghan/XPNorm/Data/data_test'

test_file='/Users/mattocallaghan/XPNorm/Data/data_red_circle'
test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle'

#test_file='/Users/mattocallaghan/XPNorm/Data/zero_bayestar'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_red_circle_20'

err_file='/Users/mattocallaghan/XPNorm/Data/err_full_ps_1'

class JaxNormFlow():
    def __init__(self,csv_location=data_file,resample=32,*args, **kwargs):


        '''
Learning the extinction map

        '''
#############################################################################
###################### DATA IMPORT #######################################
#############################################################################

        self.method='train'

        self.data=pd.read_csv(csv_location)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag', 'z_mean_psf_mag', 'y_mean_psf_mag','ks_m']]
        self.mu_error=pd.read_csv(err_file)['mu_error']
        self.data=self.data[self.mu_error/self.data['mu']<0.01]
        self.data=self.data[10**((self.data['mu']+5)/5)<4000]
        self.mu_error=self.mu_error.loc[self.data.index]
        self.data=self.data[self.mu_error/self.data['mu']<0.01]



        self.data=self.data[self.data['phot_g_mean_mag']>13]

        self.data=self.data[self.data['ks_m']-self.data['mu']<10.0]
        self.data=self.data[self.data['phot_bp_mean_mag']-self.data['phot_rp_mean_mag']>0.0]


        self.b=pd.read_csv(csv_location)[['ra','dec']].loc[self.data.index]
        self.data=self.data.values




        
        
        galactic_coord = SkyCoord(ra=self.b['ra'].values*u.degree, dec=self.b['dec'].values*u.degree, frame='icrs')
        galactic_coord = galactic_coord.transform_to('galactic')
        self.l=galactic_coord.l.value
        self.b=galactic_coord.b.value

        self.data_test=pd.read_csv(test_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']].values





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
        
        g = jnp.array([0.7, 0.95])
        bp =  jnp.array([0.97, 1.28])
        rp =  jnp.array([0.55, 0.69])
        j =  jnp.array([0.339])
        h =  jnp.array([0.2544])
        ks =  jnp.array([0.193])
        #self.extinction_vector=(jnp.einsum('ij,j->i',self.data_transform,
        #    jnp.array([0,g.mean(), bp.mean(), rp.mean(), j.mean(), h.mean(), ks.mean()])))


        self.data=jnp.einsum('ij,bj->bi',jnp.array(self.data_transform),jnp.array(self.data))

        #self.data=self.data[(self.data[:,1]<10)*(self.data[:,1]>-2)]
        self.data=self.data[(self.data[:,0]<20)]#*(self.data[:,0]>2)]

        self.data = self.data.at[:,0].set(self.data[:,0]*jnp.abs(jnp.sin(jnp.radians(self.b))))
        self.mean=jnp.mean(self.data,axis=0)
        self.std=jnp.std(self.data,axis=0)
        self.data=(self.data-self.mean)#/self.std

        self.key, self.subkey = jr.split(jr.PRNGKey(0))
        self.flow = masked_autoregressive_flow(
            self.subkey,
            base_dist=Normal(jnp.zeros(self.data.shape[1])),
            transformer=RationalQuadraticSpline(knots=10, interval=8),flow_layers=10,
        nn_width=30,)

        try:
            self.flow = eqx.tree_deserialise_leaves("/Users/mattocallaghan/XPNorm/Data/NF_Jax_dist.eqx", self.flow)
        except:
            self.train()
            eqx.tree_serialise_leaves("/Users/mattocallaghan/XPNorm/Data/NF_Jax_dist.eqx", self.flow)
            

    def train(self):

        
        self.key, self.subkey = jr.split(self.key)
        self.flow, self.losses  = fit_to_data(
                            key=self.subkey,
                            dist=self.flow,
                            x=self.data,
                            learning_rate=1e-3,
                            max_patience=1e10,
                            max_epochs=15,
                            batch_size=(2**8)
                        )
        
        

