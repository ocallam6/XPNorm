
from src.Normalising_Flow_nodist import Normalising_Flow_Trainer

import pandas as pd
import jax.numpy as jnp
import jax.random as jr


from flowjax.bijections import RationalQuadraticSpline
from flowjax.distributions import Normal
from flowjax.flows import masked_autoregressive_flow
from flowjax.tasks import two_moons
from flowjax.train import fit_to_data

from numpyro.infer import MCMC, NUTS
import numpyro.distributions as dist
import numpyro
#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data_full'
test_file='/Users/mattocallaghan/XPNorm/Data/data_test'

test_file='/Users/mattocallaghan/XPNorm/Data/data_red_circle'
test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle'

#test_file='/Users/mattocallaghan/XPNorm/Data/zero_bayestar'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_red_circle_20'

err_file='/Users/mattocallaghan/XPNorm/Data/err'

class JaxNormFlow():
    def __init__(self,csv_location=data_file,resample=32,*args, **kwargs):


        '''
Learning the extinction map

        '''
#############################################################################
###################### DATA IMPORT #######################################
#############################################################################

        self.method='train'

        self.data=pd.read_csv(csv_location)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']].values

        self.data_test=pd.read_csv(test_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']].values


        self.data_transform=jnp.array([
                                [1., 0., 0., 0., 0., 0.,0],
                                [-1., 1., 0., 0., 0., 0.,0],
                                [0., 1., -1., 0., 0., 0.,0],
                                [0., 1., 0., -1., 0., 0.,0],
                                [0., 1., 0., 0., -1., 0.,0],
                                [0., 1., 0., 0., 0., -1.,0],
                                [0., 1., 0., 0., 0., 0.,-1]])

        g = jnp.array([0.7, 0.95])
        bp =  jnp.array([0.97, 1.28])
        rp =  jnp.array([0.55, 0.69])
        j =  jnp.array([0.339])
        h =  jnp.array([0.2544])
        ks =  jnp.array([0.193])
        self.extinction_vector=(jnp.einsum('ij,j->i',self.data_transform,
            jnp.array([0,g.mean(), bp.mean(), rp.mean(), j.mean(), h.mean(), ks.mean()])))


        self.data=jnp.einsum('ij,bj->bi',jnp.array(self.data_transform),jnp.array(self.data))

        self.data=self.data[(self.data[:,1]<10)*(self.data[:,1]>-2)]
        self.data=self.data[(self.data[:,0]<20)*(self.data[:,0]>2)]
        self.data=self.data[:,1:]
        self.mean=jnp.mean(self.data,axis=0)
        self.std=jnp.std(self.data,axis=0)
        self.data=(self.data-self.mean)

        self.key, subkey = jr.split(jr.PRNGKey(0))
        self.flow = masked_autoregressive_flow(
            subkey,
            base_dist=Normal(jnp.zeros(self.data.shape[1])),
            transformer=RationalQuadraticSpline(knots=8, interval=5),)
        self.key, self.subkey = jr.split(self.key)
        self.flow, self.losses  = fit_to_data(
                            key=subkey,
                            dist=self.flow,
                            x=self.data,
                            learning_rate=1e-3,
                            max_patience=1e10,
                            max_epochs=15,
                            batch_size=2**5
                        )

