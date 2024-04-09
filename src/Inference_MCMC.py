import torch.nn as nn
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
from src.Normalising_Flow_nodist import Normalising_Flow_Trainer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import emcee
from scipy.stats import multivariate_normal

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

class Inference():
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


        self.data_transform=np.array([
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
        self.extinction_vector=torch.tensor(np.einsum('ij,j->i',self.data_transform,
            np.array([0,g.mean(), bp.mean(), rp.mean(), j.mean(), h.mean(), ks.mean()])))


        self.data=np.einsum('ij,bj->bi',np.array(self.data_transform),np.array(self.data))

        self.data=self.data[(self.data[:,1]<10)*(self.data[:,1]>-2)]
        self.data=self.data[(self.data[:,0]<20)*(self.data[:,0]>2)]
        self.data=self.data[:,1:]
        self.mean=np.mean(self.data,axis=0)
        self.std=np.std(self.data,axis=0)
        self.data=(self.data-self.mean)

        self.data_test=np.einsum('ij,bj->bi',np.array(self.data_transform),np.array(self.data_test))
        
        self.data_test=self.data_test[(self.data_test[:,1]<10)*(self.data_test[:,1]>-2)]
        self.data_test=self.data_test[(self.data_test[:,0]<20)*(self.data_test[:,0]>2)]
        self.dist_mean=self.data_test[:,0].mean()
        self.data_test[:,0]=self.data_test[:,0]-self.dist_mean
        self.data_test[:,1:]=(self.data_test[:,1:]-self.mean)
#############################################################################
###################### NEURAL NETWORK #######################################
#############################################################################




        self.nf=Normalising_Flow_Trainer()
        self.nf.load()
        self.nf.nfm.eval()
        for param in self.nf.nfm.parameters():
            param.requires_grad = False


#############################################################################
###################### Extinction Law #######################################
#############################################################################

        self.coeffs= np.array([
        [0.194059063720102, -0.000337880688254366, 0.000405004510990789, -0.000119030825664077, -2.90629429374213e-05, 9.85596051245887e-09, 1.22296149043372e-10, 0, 0, 0],
        [0.255058871064972, 7.19016588950228e-05, -0.000250455702483274, 3.99422163967702e-05, -6.83632867675118e-05, -2.3163568526324e-09, 7.26631781961228e-10, -2.27788077229475e-07, 4.97609167483581e-07, 6.67076034469308e-09],
        [0.340301468237771, -0.000826269158576803, -0.000283696380615497, 0.000103538996307887, -0.000156039957793959, 1.81587525109325e-07, 2.33928990111011e-09, 1.63733498839871e-06, 5.71693287820809e-08, 2.954302715354e-08],
        [0.663374149569189, -0.0184814428792349, 0.0100266536020118, -0.00317119320308867, -0.00649336268398495, 3.27674122186053e-05, 1.57894227641494e-06, -0.000116290469708794, 0.000117366662432525, 6.91273258513903e-06],
        [1.1516008149802, -0.0502982507379423, -0.00801054248601918, 0.0028487377407222, -0.0222045923218139, 0.000841943191161668, -1.31018008013547e-05, 0.00346423295251231, -0.000145621334026214, -6.85718568409361e-05],
        [0.993548688906439, -0.110149052160837, 0.0264447715065468, -0.00571010222810317, -0.0374363031107716, 0.00151447309438712, -2.52364537395156e-05, 0.00623824755961677, -0.000123598316318183, -0.000158499801004388]
        ])




    def extinction_coeff(self,x,a,i):

        coeff=self.coeffs[i]
        result=coeff[0]*np.ones_like(x)
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
    def log_prior(self,A0):
        if 0< A0 < 1.0:
            return 0.0
        return -np.inf

    def log_likelihood(self,x, x_obs, x_err):
        """
        Calculate the log-likelihood of a multivariate Gaussian distribution.

        Parameters:
            X (numpy.ndarray): Data points, shape (n_samples, n_features).
            mu (numpy.ndarray): Mean vector, shape (n_features,).
            sigma (numpy.ndarray): Covariance matrix, shape (n_features, n_features).

        Returns:
            float: Log-likelihood of the data.
        """
        n = x.shape[0]
        d = x.shape[1]

        constant_term = -0.5 * n * np.log(2 * np.pi)
        log_det_sigma = -0.5 * np.log(np.linalg.det(x_err))

        # Compute the quadratic term
        X_centered = x - x_obs
        quadratic_term = -0.5 * np.sum(np.dot(X_centered, np.linalg.inv(x_err)) * X_centered)

        log_likelihood = constant_term + log_det_sigma + quadratic_term
        return log_likelihood

    def log_probability(self,theta,x_obs, x_err):
        A0,x=theta[None,0],theta[None,1:]
        lp = self.nf.nfm.log_prob(torch.tensor(x)).detach().numpy()+self.log_prior(A0)
        extinction_vector=np.stack([self.extinction_coeff ((x+self.mean)[:,-1],A0,i) for i in np.arange(5,-1,-1)]).T
        extinction_vector=(np.einsum('ij,bj->bi',(self.data_transform[1:,1:]),extinction_vector))
        lp+=self.log_likelihood(x+extinction_vector*A0,x_obs,x_err)
        if not np.isfinite(lp):
            return -np.inf
            
        return lp


    def run(self,x_obs, x_err):

        pos = np.concatenate((np.zeros(x_obs[:,0:1].shape),x_obs),axis=1)

        pos=1e-4 * np.random.randn(pos.shape[0]*32,pos.shape[1])

        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability, args=(x_obs, x_err)
        )
        sampler.run_mcmc(pos, 5000, progress=True);