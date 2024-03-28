import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.optim as optim

import pandas as pd
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
from torch.utils.data import DataLoader, Dataset, RandomSampler


'''
This defines the normalising flow to learn the distribution of fluxes from Gaia and photometry

'''

#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

#directory for the pheonix spectra
data_file='/Users/mattocallaghan/XPNorm/Data/data_full'
test_file='/Users/mattocallaghan/XPNorm/Data/data_test'
err_file='/Users/mattocallaghan/XPNorm/Data/err'

class Normalising_Flow_Trainer():
    def __init__(self,csv_location=data_file,resample=32,*args, **kwargs):
        '''
      Normalising flow class and trainer.
      Uses the python normalising flow package
      This is an implementation of the Real NVP NF
        '''
        #data
        #Take in all of the data
        self.how='spline'
        self.losses=[]
        self.method='train'
        if(self.method=='infer'):
            self.data=pd.read_csv(csv_location)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']]
            self.data['c']=self.data['mu'].copy()*0
            self.data=self.data[['c','mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']]

            self.data_test=pd.read_csv(test_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']]
            self.data_test['c']=self.data_test['mu'].copy()*0+1
            self.data_test=self.data_test[['c','mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']]

            self.data=pd.concat([self.data,self.data_test],axis=0).values

            self.data_transform=np.array([[1,0., 0., 0., 0., 0., 0.,0],
                                    [0,1., 0., 0., 0., 0., 0.,0],
                                    [0,-1., 1., 0., 0., 0., 0.,0],
                                    [0,0., 1., -1., 0., 0., 0.,0],
                                    [0,0., 1., 0., -1., 0., 0.,0],
                                    [0,0., 1., 0., 0., -1., 0.,0],
                                    [0,0., 1., 0., 0., 0., -1.,0],
                                    [0,0., 1., 0., 0., 0., 0.,-1]])

            g = np.array([0.7, 0.95])
            bp =  np.array([0.97, 1.28])
            rp =  np.array([0.55, 0.69])
            j =  np.array([0.71, 0.73])/3.1
            h =  np.array([0.45, 0.47])/3.1
            ks =  np.array([0.34, 0.36])/3.1

            self.extinction_vector=torch.tensor(np.einsum('ij,j->i',self.data_transform,
                np.array([0,0,g.mean(), bp.mean(), rp.mean(), j.mean(), h.mean(), ks.mean()])))
       

            self.data=np.einsum('ij,bj->bi',np.array(self.data_transform),np.array(self.data))

            self.data=self.data[(self.data[:,1+1]<10)*(self.data[:,1+1]>-2)]
            self.data=self.data[(self.data[:,0+1]<14)*(self.data[:,0+1]>4)]
            self.mean=np.mean(self.data,axis=0)
            self.std=np.std(self.data,axis=0)
            self.data[:,1:]=(self.data-self.mean)[:,1:]
        else:
            self.data=pd.read_csv(csv_location)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']]


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
            self.data=self.data[(self.data[:,0]<20)*(self.data[:,0]>2)]

            self.data=self.data[:,1:]
            self.mean=np.mean(self.data,axis=0)
            self.std=np.std(self.data,axis=0)
            self.data=(self.data-self.mean)
        # is this right



        #print(self.data.shape)
        # Depth of the Neural Network
        self.K=5
        # Number of features
        self.latent_size=self.data.shape[1]
        # Mask
        # Training Parameters
        self.num_epochs=15
        self.lr=1e-3
        self.batch_size=2**5#self.data.shape[0]
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
            

            if(self.method=='infer'):

                b = torch.Tensor([0,1,1,1,1,1,1,1])
                s = nf.nets.MLP([self.latent_size, 2 * self.latent_size, self.latent_size], init_zeros=True,output_fn='sigmoid')
                t = None#nf.nets.MLP([self.latent_size, 2 * self.latent_size, self.latent_size], init_zeros=True)
                self.flows += [nf.flows.MaskedAffineFlow(b, t, s)]

                

                b = torch.Tensor([1,1,0,0,0,0,0,0])
                s = None#nf.nets.MLP([self.latent_size, 2 * self.latent_size, self.latent_size], init_zeros=True)
                self.t = nf.nets.MLP([self.latent_size-1, 10, self.latent_size], init_zeros=True)
                self.flows += [CustomMaskedAffineFlow(b, self.t, s,self.extinction_vector)]
                #flows += [nf.flows.ActNorm(self.latent_size)]

        # Target distribution - Multivaritate Gaussian
        self.q0 = nf.distributions.DiagGaussian(self.latent_size,trainable=False)
        #self.q0 = nf.distributions.GaussianMixture(n_modes=2,dim=self.latent_size,trainable=False)

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

        
    def group_shuffle_batch(self,data_tensor, batch_size):
        num_chunks = len(data_tensor) // 32
        chunks = [data_tensor[i*32:(i+1)*32] for i in range(num_chunks)]
        shuffled_chunks = torch.randperm(num_chunks).tolist()
        shuffled_chunks = [chunks[i] for i in shuffled_chunks]

        batched_data = [shuffled_chunks[i:i+batch_size] for i in range(0, num_chunks, batch_size)]
        return batched_data


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
            data_tensor=torch.tensor(self.data,dtype=torch.float32).to(self.device)
        else:
            data_tensor=torch.tensor(self.data,dtype=torch.double).to(self.device)



# Example usage

        batched_data = self.group_shuffle_batch(data_tensor, batch_size=self.batch_size)
        #Dataloader object
        training_loader = torch.utils.data.DataLoader(data_tensor, batch_size=self.batch_size, shuffle=True)
        count=1
        #Iterate over the data.
        for _, data in enumerate(training_loader):
        #for data in batched_data:
            #data=torch.concat(data,axis=0)
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
            self.losses.append(loss.item()/self.batch_size)
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
            self.losses.append(avg_loss)
            print('Average Loss {}:'.format(avg_loss))

            if(avg_loss<running_loss_best):
                print('save')
                running_loss_best=avg_loss        
                #torch.save(self.nfm.state_dict(), '/Users/mattocallaghan/XPNorm/Data/north_south_nf_original')
                torch.save(self.nfm.state_dict(), '/Users/mattocallaghan/XPNorm/Data/north_south_nf_nodistance')

        self.nfm.eval()

    def load(self):

        self.nfm.load_state_dict(torch.load('/Users/mattocallaghan/XPNorm/Data/north_south_nf_nodistance'))
        #self.nfm.load_state_dict(torch.load('/Users/mattocallaghan/XPNorm/Data/north_south_nf_infer'))

        self.nfm.eval()
    def sample(self,num_samples):
        test=self.nfm.sample(num_samples=num_samples)[0].detach().cpu().numpy()
        return test+self.mean



class CustomMaskedAffineFlow(nf.flows.MaskedAffineFlow):
    def __init__(self, b, t, s,extinction_vector):
        super().__init__(b, t, s)
        self.extinction_vector=extinction_vector
        self.ebv= torch.nn.Parameter(torch.tensor([1.0]),requires_grad=True)

        
    def forward(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.ebv*(z_masked[:,0][:,None])#self.t(z_masked[:,1:])*(z_masked[:,0][:,None])
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z * torch.exp(scale) + self.extinction_vector*trans)
        log_det = torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det


    def inverse(self, z):
        z_masked = self.b * z
        scale = self.s(z_masked)
        nan = torch.tensor(np.nan, dtype=z.dtype, device=z.device)
        scale = torch.where(torch.isfinite(scale), scale, nan)
        trans = self.ebv*(z_masked[:,0][:,None])#self.t(z_masked[:,1:])*(z_masked[:,0][:,None])
        trans = torch.where(torch.isfinite(trans), trans, nan)
        z_ = z_masked + (1 - self.b) * (z - self.extinction_vector*trans) * torch.exp(-scale)
        log_det = -torch.sum((1 - self.b) * scale, dim=list(range(1, self.b.dim())))
        return z_, log_det