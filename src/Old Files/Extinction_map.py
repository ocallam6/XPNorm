
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
from src.Normalising_Flow import Normalising_Flow_Trainer
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn

'''
This defines a Neural Network to learn the extinction law generated from synthetic spectra
'''

#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data_full'
test_file='/Users/mattocallaghan/XPNorm/Data/data_test'
err_file='/Users/mattocallaghan/XPNorm/Data/err'


class Extinction_Map(nn.Module):

    """
    This class defines the extinction map. Takes in
    """
    def __init__(self,*args, **kwargs):
        super(Extinction_Map, self).__init__()

        self.fc1 = torch.nn.Linear(1, 50)  # Input size: 2, Output size: 4
        self.fc2 = torch.nn.Linear(50, 10) 
        self.fc_out = torch.nn.Linear(10, 1) 

        self.nf=Normalising_Flow_Trainer()
        self.nf.load()
        self.nf.nfm.eval()

    def forward(self,x):
        ebv = torch.relu(self.fc1(x[:,0][:,None]))
        ebv = torch.relu(self.fc2(ebv))
        ebv= torch.sigmoid(self.fc_out(ebv))
        x=x-self.nf.extinction_vector[None,:]*ebv

        log_q = torch.zeros(len(x), device=x.device)
        z = x
        self.nf.nfm.eval()
        for i in range(len(self.nf.flows) - 1, -1, -1):
            z, log_det = self.nf.flows[i].inverse(z)
            log_q += log_det

    
        log_q += self.nf.q0.log_prob(z)
        return -torch.mean(log_q)

    def test_ebv(self,x,ebv):
        z=x-self.nf.extinction_vector[None,:]*ebv

        log_q = torch.zeros(len(z), device=z.device)
        self.nf.nfm.eval()
        for i in range(len(self.nf.flows) - 1, -1, -1):
            z, log_det = self.nf.flows[i].inverse(z)
            log_q += log_det

    
        log_q += self.nf.q0.log_prob(z)
        return -torch.mean(log_q)


class Extinction_Trainer():
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
        j =  np.array([0.71, 0.73])/3.1
        h =  np.array([0.45, 0.47])/3.1
        ks =  np.array([0.34, 0.36])/3.1

        self.extinction_vector=torch.tensor(np.einsum('ij,j->i',self.data_transform,
            np.array([0,g.mean(), bp.mean(), rp.mean(), j.mean(), h.mean(), ks.mean()])))
    

        self.data=np.einsum('ij,bj->bi',np.array(self.data_transform),np.array(self.data))

        self.data=self.data[(self.data[:,1]<10)*(self.data[:,1]>-2)]
        self.data=self.data[(self.data[:,0]<14)*(self.data[:,0]>4)]
        self.mean=np.mean(self.data,axis=0)
        self.std=np.std(self.data,axis=0)
        self.data=(self.data-self.mean)

        self.data_test=np.einsum('ij,bj->bi',np.array(self.data_transform),np.array(self.data_test))

        self.data_test=self.data_test[(self.data_test[:,1]<10)*(self.data_test[:,1]>-2)]
        self.data_test=self.data_test[(self.data_test[:,0]<14)*(self.data_test[:,0]>4)]
  
        self.data_test=(self.data_test-self.mean)

#############################################################################
###################### NEURAL NETWORK #######################################
#############################################################################

        self.model=Extinction_Map()

        self.num_epochs=15
        self.lr=1e-3
        self.batch_size=2**5#self.data.shape[0]
        self.enable_cuda = True
        self.device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() and self.enable_cuda else 'cpu')
        self.model = self.model.to(self.device)
        if(self.device!='mps'): #mac testing
            self.model = self.model.double()
        self.optimizer= optim.Adam(self.model.parameters(),lr=self.lr, weight_decay=1e-5)


    def one_epoch_extinction(self):
        '''
        Our model will train an epoch by iterating over each dask partition.
        '''
        #Running loss of the training at a given epoc
        running_loss=0.0


        if(self.device=='mps'):
            data_tensor=torch.tensor(self.data_test,dtype=torch.float32).to(self.device)
        else:
            data_tensor=torch.tensor(self.data_test,dtype=torch.double).to(self.device)

        training_loader = torch.utils.data.DataLoader(data_tensor, batch_size=self.batch_size, shuffle=True)
        count=1
        #Iterate over the data.
        for _, data in enumerate(training_loader):
        #for data in batched_data:
            #data=torch.concat(data,axis=0)
            #Zero the gradients for each batch learning
            self.optimizer.zero_grad()
            #Evaluate the loss
            loss = self.model(data)
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                #back propegation
                loss.backward()
                #update weights
                self.optimizer.step()
            running_loss += loss.item()

            #delete from memory to save RAM
            
            del(data, loss)
            count+=1
     
        return running_loss
    def train(self):
        running_loss_best=1e16
        for epoch in range(self.num_epochs):

            print('EPOCH {}:'.format(epoch + 1))
            self.model.train(True)
            avg_loss = self.one_epoch_extinction()
            print('Average Loss {}:'.format(avg_loss))

            if(avg_loss<running_loss_best):
                print('save')
                running_loss_best=avg_loss        
                torch.save(self.model.state_dict(), '/Users/mattocallaghan/XPNorm/Data/extinction_map')
            ebv = torch.relu(self.model.fc1(torch.tensor(self.data_test)[:,0:1]))
            ebv = torch.relu(self.model.fc2(ebv))
            ebv= torch.sigmoid(self.model.fc_out(ebv))
            distance=10**((self.data_test[:,0]+self.mean[0]+5)/5)
            plt.hist2d(distance,ebv[:,0].detach().numpy())
            plt.scatter(distance,ebv[:,0].detach().numpy(),c='black',s=1)

            plt.show()

        self.model.eval()


