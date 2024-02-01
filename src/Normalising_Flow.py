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
'''
This defines the normalising flow to learn the distribution of fluxes from Gaia and photometry

'''

#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data'

class Normalising_Flow_Trainer():
    def __init__(self,csv_location=data_file,*args, **kwargs):
        '''
      Normalising flow class and trainer.
      Uses the python normalising flow package
      This is an implementation of the Real NVP NF
        '''
        #data
        self.how='spline'

        self.data=pd.read_csv(csv_location)[['parallax','ks_m' , 'phot_g_mean_mag', 'j_m', 'h_m']]
        self.data=self.data[self.data['parallax']>0].reset_index(drop=True).values
        for i in range(2,self.data.shape[1]):
            self.data[:,i]=self.data[:,i]-self.data[:,1]
        self.data[:,1]=self.data[:,1]-2.5*np.log10(((1000/(self.data[:,0]))/10)**2)
        
        self.mean=np.mean(self.data,axis=0)
        
        self.std=np.std(self.data,axis=0)
        

        self.data=(self.data-self.mean)


        #print(self.data.shape)
        # Depth of the Neural Network
        self.K=64
        # Number of features
        self.latent_size=self.data.shape[1]-1
        # Mask
        # Training Parameters
        self.num_epochs=100
        self.lr=1e-3
        self.batch_size=2**8#self.data.shape[0]
        self.loss_hist=np.array([])
        # Flow layers
        if(self.how=='realnvp'):
            self.b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(self.latent_size)])
            
            self.flows = []
            for i in range(self.K):
                s = nf.nets.MLP([self.latent_size, 2 * self.latent_size, self.latent_size], init_zeros=True)
                t = nf.nets.MLP([self.latent_size, 2 * self.latent_size, self.latent_size], init_zeros=True)
                if i % 2 == 0:
                    self.flows += [nf.flows.MaskedAffineFlow(self.b, t, s)]
                else:
                    self.flows += [nf.flows.MaskedAffineFlow(1 - self.b, t, s)]
                self.flows += [nf.flows.ActNorm(self.latent_size)]
        if(self.how=='spline'):
            self.flows = []
            for i in range(self.K):
                self.flows += [nf.flows.AutoregressiveRationalQuadraticSpline(self.latent_size, 2*self.latent_size, 2)]
                self.flows += [nf.flows.LULinearPermute(self.latent_size)]


        # Target distribution - Multivaritate Gaussian
        self.q0 = nf.distributions.DiagGaussian(self.latent_size,trainable=False)

        # Model definition
        self.nfm = nf.NormalizingFlow(q0=self.q0, flows=self.flows)

        # Meta Parameters - For Learning
        self.enable_cuda = True
        self.device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() and self.enable_cuda else 'cpu')
        self.nfm = self.nfm.to(self.device)
        if(self.device!='mps'): #mac testing
            self.nfm = self.nfm.double()
        self.optimizer= optim.Adam(self.nfm.parameters(),lr=self.lr, weight_decay=1e-5)



        #Initialize ActNorm
        #self.z, _ = self.nfm.sample(num_samples=2 ** 7)

        



    def train_one_epoch(self):
        '''
        Our model will train an epoch by iterating over each dask partition.
        '''
        #Running loss of the training at a given epoc
        running_loss=0.0
        #Load in the partition
        #Apply the data transforms. This is a custom piece of code in the pipeline and needs to match the transformation 
        #in the transformation section.

        if(self.device=='mps'):
            data_tensor=torch.tensor(self.data[:,1:],dtype=torch.float32).to(self.device)
        else:
            data_tensor=torch.tensor(self.data[:,1:],dtype=torch.double).to(self.device)

        #Dataloader object
        training_loader = torch.utils.data.DataLoader(data_tensor, batch_size=self.batch_size, shuffle=True)
        count=1
        #Iterate over the data.
        for _, data in enumerate(training_loader):
            if(count%50==0):
                print(running_loss/count)
            #Zero the gradients for each batch learning
            self.optimizer.zero_grad()
            #Evaluate the loss
            loss = self.nfm.forward_kld(data)
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
            self.nfm.train(True)
            avg_loss = self.train_one_epoch()
            if(avg_loss<running_loss_best):
                running_loss_best=avg_loss        
                torch.save(self.nfm.state_dict(), '/Users/mattocallaghan/XPNorm/Data/model_train_gaia_parallax_c')

            print('Average Loss {}:'.format(avg_loss))
            self.loss_hist = np.append(self.loss_hist, np.array([avg_loss]))

        self.nfm.eval()

    def load(self):

        self.nfm.load_state_dict(torch.load('/Users/mattocallaghan/XPNorm/Data/model_train_gaia_parallax_c'))
        self.nfm.eval()
    def sample(self,num_samples):
        test=self.nfm.sample(num_samples=num_samples)[0].detach().cpu().numpy()
        return test+self.mean[1:]

