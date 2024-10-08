"""
We learn the extinction law from the Phoenix spectra as a function of A0, and two colors


Learn it as a Gaussian model. This is because of incertainty in modelling at lower absolute magnitude. The extinction coeffieicents are driven
by feh here. 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

in_file='/Users/mattocallaghan/XPNorm/Data/exts.npy'
in_file_2='/Users/mattocallaghan/XPNorm/Data/vals.npy'

TRAIN=False
PLOT=True

class Data_Extinction():
    def __init__(self):


        '''
Learning the extinction map. We generate the data from the models. 
This is to be learned as a function of A0, GK, RPK

        '''
        ################################################################################################
        ###################### Training Data #######################################
        ##########################################################

        # Data that has been generated from the extinciton law.
        data_file=in_file # Training
        data_file_2=in_file_2 # Test

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
        self.data=pd.DataFrame(np.concatenate((teff,logg,feh,av,np.load(data_file),np.load(data_file_2)),1),columns=columns)
        self.data=self.data[['teff','logg','feh','av',"Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3','2MASS_J','2MASS_H','2MASS_Ks','PS_g','PS_i','PS_r','PS_y','PS_z',"a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z']].dropna()
        
        self.teff=self.data['teff'].values
        self.feh=self.data['feh'].values
        self.logg=self.data['logg'].values

        self.data=self.data[['av',"Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3','2MASS_J','2MASS_H','2MASS_Ks','PS_g','PS_i','PS_r','PS_y','PS_z',"a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z']].dropna()

        self.data[["k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']]=(self.data[["a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z']].values)/self.data['av'].values[:,None]
        self.data['GK']=self.data['Gaia_G_EDR3']-self.data['2MASS_Ks']
        self.data['RPK']=self.data['Gaia_RP_EDR3']-self.data['2MASS_Ks']
        self.bprp=self.data['av']#-self.data['Gaia_G_EDR3']


        #self.data=self.data[['av',"GK","RPK","k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']].dropna().values
        self.data=self.data[['av',"GK","RPK","k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']].dropna().values




        self.data_transform=np.array([
                                [1., 0., 0., 0., 0., 0.,0],
                                [0., 0., 0., 0., 0., 0.,1],
                                [0., 0., 1., 0., 0., 0.,-1],
                                [0., 0., 0., 1., 0., 0.,-1],
                                [0., 0., 0., 0., 1., 0.,-1],
                                [0., 0., 0., 0., 0., 1.,-1],
                                [0., 1., 0., 0., 0., 0.,-1]])
        


        self.mean=np.mean(self.data,axis=0)
        self.std=np.std(self.data,axis=0)
        self.data=(self.data-self.mean)/self.std

        ################################################################################################
        ###################### Test Data #######################################
        ##########################################################


        teff=np.arange(3500,6000,50)
        logg=np.arange(3,5,0.05)
        feh=np.arange(-1,0.5,0.1)
        #Rv=np.arange(1,5,0.2)
        av=np.arange(0.001,1,0.05)

        mesh=np.meshgrid(teff,logg,feh,av)
        teff=mesh[0].flatten()[:,None]
        logg=mesh[1].flatten()[:,None]
        feh=mesh[2].flatten()[:,None]
        av=mesh[3].flatten()[:,None]
        columns = ['teff','feh','logg','av',"a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_WISE_W1','a_WISE_W2','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z',"Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3','2MASS_J','2MASS_H','2MASS_Ks','WISE_W1','WISE_W2','PS_g','PS_i','PS_r','PS_y','PS_z'] #add wise on later.
        self.data_test=pd.DataFrame(np.concatenate((teff,logg,feh,av,np.load('/Users/mattocallaghan/XPNorm/Data/exts_test.npy'),np.load('/Users/mattocallaghan/XPNorm/Data/vals_test.npy')),1),columns=columns)
        self.data_test=self.data_test[['av',"Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3','2MASS_J','2MASS_H','2MASS_Ks','PS_g','PS_i','PS_r','PS_y','PS_z',"a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z']].dropna()

        self.data_test[["k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']]=(self.data_test[["a_Gaia_G_EDR3", "a_Gaia_BP_EDR3", 'a_Gaia_RP_EDR3','a_2MASS_J','a_2MASS_H','a_2MASS_Ks','a_PS_g','a_PS_i','a_PS_r','a_PS_y','a_PS_z']].values)/self.data_test['av'].values[:,None]
        self.data_test['GK']=self.data_test['Gaia_G_EDR3']-self.data_test['2MASS_Ks']
        self.data_test['RPK']=self.data_test['Gaia_RP_EDR3']-self.data_test['2MASS_Ks']

        #self.data_test=self.data_test[['av',"GK","RPK","k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']].dropna().values
        self.data_test=self.data_test[['av',"GK","RPK","k_Gaia_G_EDR3", "k_Gaia_BP_EDR3", 'k_Gaia_RP_EDR3','k_2MASS_J','k_2MASS_H','k_2MASS_Ks','k_PS_g','k_PS_i','k_PS_r','k_PS_y','k_PS_z']].dropna().values
        
class Extinction_NN(nn.Module):
    """
    A0,GK,RPK ---> Extinction coefficients
    """

    def __init__(self):
        super(Extinction_NN, self).__init__()
        self.fc1 = nn.Linear(3, 100)  # Input dimension 12, output dimension 64
        self.fc2 = nn.Linear(100, 100)  # Input dimension 64, output dimension 32
        self.fc3 = nn.Linear(100, 11)  # Input dimension 32, output dimension 11
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if(TRAIN):
    # Generate Training Data
    ext=Data_Extinction()
    model = Extinction_NN()

    # Define the optimizer (Adam optimizer)
    optimizer = optim.Adam(model.parameters())

    # Define the loss function (Mean Squared Error)
    criterion = nn.MSELoss()

    # You can add L2 regularization by adding weight decay parameter to Adam optimizer
    #optimizer = optim.Adam(model.parameters(), weight_decay=0.001)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.001)





    # Convert numpy arrays to PyTorch tensors
    x_input = torch.tensor(ext.data, dtype=torch.float32)

    # Create a TensorDataset
    dataset = TensorDataset(x_input)

    # Define batch size
    batch_size = 2**4

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=False)

    # Define the model, optimizer, and loss function (assuming you already defined them as shown in the previous example)
    outputs = model(x_input[::1000,:3])

    # Train the model for 50 epochs


    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs = data[0]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs[:,:3])

            # Calculate loss
            loss = criterion(outputs, inputs[:,3:])

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 0:  # Print every 10 mini-batches
                print('[%d, %5d] loss: %.10f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    torch.save(model.state_dict(), '/Users/mattocallaghan/XPNorm/Data/model_extinction_2')

if(PLOT):

    ext=Data_Extinction()

    model = Extinction_NN()  # Re-instantiate the model
    model.load_state_dict(torch.load('/Users/mattocallaghan/XPNorm/Data/model_extinction_2'))


    fig,axes=plt.subplots(1,1)
    axes=[axes]#.flatten()
    axes[0].scatter((ext.data[:,2]*ext.std[2]+ext.mean[2]),(ext.data[:,1]*ext.std[1]+ext.mean[1]),c=(ext.data[:,3:][:,1]*ext.std[3+1]+ext.mean[3+1]),s=0.1)
    axes[0].set_xlabel('$m_{G}-m_{K_s}$', fontsize=12)  # Adjust the label and font size as needed
    axes[0].set_ylabel('$A_G/A_V$', fontsize=12) 
    plt.show()

    fig,axes=plt.subplots(3,1)
    axes=axes.flatten()
    axes[0].scatter((ext.data[:,1]*ext.std[1]+ext.mean[1]),(ext.data[:,3:][:,1]*ext.std[3+1]+ext.mean[3+1]),c=ext.logg,s=0.1)
    axes[0].set_xlabel('$m_{G}-m_{K_s}$', fontsize=12)  # Adjust the label and font size as needed
    axes[0].set_ylabel('$A_G/A_V$', fontsize=12)  # Adjust the label and font size as needed

    axes[1].scatter((ext.data[:,1]*ext.std[1]+ext.mean[1]),(ext.data[:,3:][:,1]*ext.std[3+1]+ext.mean[3+1]),c=ext.feh,s=0.1)
    axes[1].set_xlabel('$m_{G}-m_{K_s}$', fontsize=12)  # Adjust the label and font size as needed
    axes[1].set_ylabel('$A_G/A_V$', fontsize=12)  # Adjust the label and font size as needed

    axes[2].scatter((ext.data[:,1]*ext.std[1]+ext.mean[1]),(ext.data[:,3:][:,1]*ext.std[3+1]+ext.mean[3+1]),c=ext.teff,s=0.1)
    axes[2].set_xlabel('$m_{G}-m_{K_s}$', fontsize=12)  # Adjust the label and font size as needed
    axes[2].set_ylabel('$A_G/A_V$', fontsize=12)  # Adjust the label and font size as needed

    plt.tight_layout()
    plt.show()


    #plt.scatter(outputs[:,0],(x_input[:,3:].detach().numpy()*ext.std[3:]+ext.mean[3:])[:,0])

    outputs=model(torch.tensor(ext.data[:,:3],dtype=torch.float32)).detach().numpy()*ext.std[3:]+ext.mean[3:]
    plt.scatter((ext.data[:,1]*ext.std[1]+ext.mean[1]),(ext.data[:,3:][:,2]*ext.std[3+2]+ext.mean[3+2]),label='real data')
    plt.scatter(ext.data[:,1]*ext.std[1]+ext.mean[1],outputs[:,2],label='data output')
    test=torch.tensor((ext.data_test-ext.mean)/ext.std,dtype=torch.float32)

    outputs = model(test[:,:3]).detach().numpy()*ext.std[3:]+ext.mean[3:]
    
    plt.scatter((ext.data_test[:,1]),(ext.data_test[:,3:][:,2]),label='real test')
    plt.scatter(ext.data_test[:,1],outputs[:,2], label='output test')
    plt.legend()
    plt.show()



    test=torch.tensor((ext.data_test-ext.mean)/ext.std,dtype=torch.float32)

    outputs = model(test[:,:3]).detach().numpy()*ext.std[3:]+ext.mean[3:]
    
    plt.scatter((ext.data_test[:,1]),(ext.data_test[:,3:][:,2]))
    plt.scatter(ext.data_test[:,1],outputs[:,2])
    plt.show()

    plt.plot(((ext.data_test[:,3:])[:,1]-outputs[:,1]))
    print(np.std((ext.data_test[:,3:][:,1]-outputs[:,1])))
    plt.show()

    plt.scatter((ext.data[:,1]*ext.std[1]+ext.mean[1]),(ext.data[:,3:][:,2]*ext.std[3+2]+ext.mean[3+2]))
    plt.scatter((ext.data_test[:,1]),(ext.data_test[:,3:][:,2]))


    plt.show()





