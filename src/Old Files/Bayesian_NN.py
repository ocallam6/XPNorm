"""
THIS IS AN OBSELETE FILE
This is an old method of using Bayesian Neural Networks to derive extinction
profiles along the line of sight.


"""

import torch.nn as nn
import pandas as pd
import numpy as np
import torchbnn as bnn
import torch
import torch.optim as optim
from src.Normalising_Flow import Normalising_Flow_Trainer
import matplotlib.pyplot as plt
from tqdm import tqdm


#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data_full'
test_file='/Users/mattocallaghan/XPNorm/Data/data_test'
err_file='/Users/mattocallaghan/XPNorm/Data/err'

class BayesianExtinction_Trainer():
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

        self.model = nn.Sequential(
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=1, out_features=100),
                nn.ReLU(),
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=100),
                nn.ReLU(),
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=100, out_features=1),
                nn.ReLU()

            )


        self.nf=Normalising_Flow_Trainer()
        self.nf.load()
        self.nf.nfm.eval()
        for param in self.nf.nfm.parameters():
            param.requires_grad = False

        self.num_epochs=10
        self.lr=1e-3
        self.batch_size=2**5#self.data.shape[0]
        self.enable_cuda = True
        self.device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() and self.enable_cuda else 'cpu')
        self.model = self.model.to(self.device)
        if(self.device!='mps'): #mac testing
            self.model = self.model.double()
        self.optimizer= optim.Adam(self.model.parameters(),lr=self.lr, weight_decay=1e-5)

        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = 0.1



    def loss(self,ebv,x):

        x=x-self.nf.extinction_vector[None,:]*ebv

        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.nf.flows) - 1, -1, -1):
            z, log_det = self.nf.flows[i].inverse(z)
            log_q += log_det


        log_q += self.nf.q0.log_prob(z)
        return -torch.mean(log_q)


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
        epoch_loss = 0.0
        epoch_kl = 0.0
        for _, data in enumerate(training_loader):
            self.optimizer.zero_grad()

            ebv = self.model(data[:, 0:1])
            n_log_l = self.loss(ebv, data)
            kl = self.kl_loss(self.model)
            loss = n_log_l + self.kl_weight * kl

            loss.backward()
            self.optimizer.step()

            del(data)
            
            epoch_loss += n_log_l.item()
            epoch_kl += kl.item()

            # Update tqdm progress bar
        tqdm.write(f'Avg log_likelihood : {epoch_loss:.2f}, Avg KL : {epoch_kl:.2f}')


        return running_loss+1
    def train(self):
        running_loss_best=1e16
        for epoch in range(self.num_epochs):

            print('EPOCH {}:'.format(epoch + 1))
            self.model.train(True)
            avg_loss = self.one_epoch_extinction()
            print('Average Loss {}:'.format(avg_loss))

            #if(avg_loss<running_loss_best):
            #    print('save')
            #    running_loss_best=avg_loss        
            torch.save(self.model.state_dict(), '/Users/mattocallaghan/XPNorm/Data/BNN_RElu_pasiphae_original')
            #ebv=self.model(torch.tensor(self.data_test[:,0:1])).detach().numpy()
            #distance=10**((torch.tensor(self.data_test[:,0:1]).detach().numpy()+self.mean[0]+5)/5)
            #plt.hist2d(distance[:,0],ebv[:,0])
            #plt.scatter(distance[:,0],ebv[:,0],s=1)

            #plt.show()

        self.model.eval()

    def load(self):

        self.model.load_state_dict(torch.load('/Users/mattocallaghan/XPNorm/Data/BNN_RElu_pasiphae_original'))
        self.model.eval()