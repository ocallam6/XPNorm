
from turtle import forward
import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import dask.dataframe as dd
import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch import nn
from Normalising_Flow_nodist import Normalising_Flow_Trainer
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.distributions import Distribution, constraints
from pyro.infer import MCMC, NUTS

'''
HMC 

'''

#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data_full'
test_file='/Users/mattocallaghan/XPNorm/Data/data_test'
err_file='/Users/mattocallaghan/XPNorm/Data/err'
normalising_flow=Normalising_Flow_Trainer()

def model(obs):
    # Prior for A0
    A0 = pyro.sample("A0", dist.Uniform(0, 1))
    extinction= torch.tensor(normalising_flow.extinction_vector[None,1:])*A0

    x=pyro.sample("x",NF_Dist(),obs=obs+extinction)
    # Prior for A1 to AN

    obs = pyro.sample("obs", dist.MultivariateNormal(x+extinction, (0.05**2)*torch.diag(torch.ones(6))), obs=obs)

def run():
    data_test=pd.read_csv(test_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']].values


    data_transform=np.array([
                            [1., 0., 0., 0., 0., 0.,0],
                            [-1., 1., 0., 0., 0., 0.,0],
                            [0., 1., -1., 0., 0., 0.,0],
                            [0., 1., 0., -1., 0., 0.,0],
                            [0., 1., 0., 0., -1., 0.,0],
                            [0., 1., 0., 0., 0., -1.,0],
                            [0., 1., 0., 0., 0., 0.,-1]])

    g = np.array([0.7, 0.95])
    bp =  np.array([0.97, 1.28])
    rp =  np.array([0.55, 0.69])
    j =  np.array([0.339])
    h =  np.array([0.2544])
    ks =  np.array([0.193])
    extinction_vector=torch.tensor(np.einsum('ij,j->i',data_transform,
        np.array([0,g.mean(), bp.mean(), rp.mean(), j.mean(), h.mean(), ks.mean()])))


    data_test=np.einsum('ij,bj->bi',np.array(data_transform),np.array(data_test))
    
    data_test=data_test[(data_test[:,1]<10)*(data_test[:,1]>-2)]
    data_test=data_test[(data_test[:,0]<20)*(data_test[:,0]>2)]
    dist_mean=data_test[:,0].mean()
    data_test[:,0]=data_test[:,0]-dist_mean
    data_test[:,1:]=(data_test[:,1:]-normalising_flow.mean)


    nuts_kernel = NUTS(model)

    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(data_test[100:101,:])

class NF_Dist(Distribution):
    support = constraints.real
    def __init__(self):
        self.normalising_flow=Normalising_Flow_Trainer()
        self.normalising_flow.load()
        self.normalising_flow.nfm.eval()
        self.normalising_flow.nfm=self.normalising_flow.nfm.double()
        for param in self.normalising_flow.nfm.parameters():
            param.requires_grad = False
        super().__init__()

    def sample(self, key=(), sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        return torch.tensor(self.normalising_flow.nfm.log_prob(value))

run()