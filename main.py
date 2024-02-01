from src import Normalising_Flow
from dustmaps.lenz2017 import Lenz2017Query
import dustmaps
from astropy.coordinates import SkyCoord,Galactic
from dustmaps.sfd import SFDQuery
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dustmaps.bayestar import BayestarQuery


if __name__=="__main__":
    nf=Normalising_Flow.Normalising_Flow_Trainer()




'''
    nf.load()
    extinction=np.array([0.306,2.69-0.306,0.717-0.306,0.464-0.306])
    sfd = SFDQuery()

    c_med = SkyCoord(304.303, -32.6786, unit="deg")
    test_data=pd.read_csv('/Users/mattocallaghan/XPNorm/Data/data')[['parallax','ks_m' , 'phot_g_mean_mag', 'j_m', 'h_m']]
    test_data=test_data[:len(test_data)//10]
    test_data=test_data[test_data['parallax']>0].reset_index(drop=True).values
    for i in range(2,test_data.shape[1]):
        test_data[:,i]=test_data[:,i]-test_data[:,1]
    test_data[:,1]=test_data[:,1]-2.5*np.log10(((1000/(test_data[:,0]))/10)**2)

    test_error=pd.read_csv('/Users/mattocallaghan/XPNorm/Data/err_test')[['ks_msigcom' , 'g_error', 'j_msigcom', 'h_msigcom']].dropna().values.mean()*2

    test_data_scaled=(test_data-nf.mean)
    logs = []
    for i in np.arange(-0.3, 0.3, 0.01):
        logs.append(nf.nfm.forward_kld(torch.tensor(test_data_scaled[:, 1:]) - extinction * i).mean().detach().numpy())
    E=np.exp(-1*(1/(2*test_error))*np.array(logs))
    print((E/E.sum()).mean())
    print((E/E.sum()).std())
    plt.plot(np.arange(-0.3, 0.3, 0.01),E/E.sum() )
    plt.xlabel('E(B-V)')  # Add a label for the x-axis
    plt.ylabel('P(E(B-V))')  # Add a label for the y-axis
    plt.title('304.303, -59.6786')  # Add a title for the plot
    text = 'Planck E(B-V)= '+str(sfd.query(c_med))
    plt.annotate(text, xy=(1, 1), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round', alpha=0.3, facecolor='white'))


    # Adjust the layout for a better appearance
    plt.tight_layout()

    # Save the figure to a file (e.g., in PNG format)
    plt.savefig('low_extinction.png')

    # Show the plot
    plt.show()


    nf.load()
    # Extinction Vector
    extinction=np.array([0.306,2.69-0.306,0.717-0.306,0.464-0.306])

    #SFD Dust Map

    sfd = SFDQuery()
    c_hi_lat = SkyCoord(30.303, 32.6786, unit="deg").transform_to(Galactic())

    # Note that below, we could use version='bayestar2017' to get the newer
    # version of the map. Note, however, that the reddening units are not
    # identical in the two versions of the map. See Green et al. (2018) for
    # an explanation of the units.
    #bayestar = BayestarQuery()

    #ebv = bayestar(c_hi_lat)
    test_data=pd.read_csv('/Users/mattocallaghan/XPNorm/Data/data_test_bayestar')[['parallax','ks_m' , 'phot_g_mean_mag', 'j_m', 'h_m']]
    test_data=test_data[test_data['parallax']>0].reset_index(drop=True).values
    for i in range(2,test_data.shape[1]):
        test_data[:,i]=test_data[:,i]-test_data[:,1]
    test_data[:,1]=test_data[:,1]-2.5*np.log10(((1000/(test_data[:,0]))/10)**2)
    test_data_scaled=(test_data-nf.mean)

    logs = []
    for i in np.arange(-0.3, 0.3, 0.01):
        logs.append((test_data_scaled.mean(0)[1:] - extinction * i).mean())
 

    #logs = []
    #for i in np.arange(-0.3, 0.3, 0.01):
    #    logs.append(nf.nfm.forward_kld(torch.tensor(test_data_scaled[:, 1:]) - extinction * i).mean().detach().numpy())

    plt.plot(np.arange(-0.3, 0.3, 0.01), logs)
    plt.xlabel('E(B-V)')  # Add a label for the x-axis
    plt.ylabel('P(E(B-V))')  # Add a label for the y-axis
    #plt.title('Bayestar= '+str(ebv.mean()))  # Add a title for the plot
    text = 'Planck E(B-V)= '+str(sfd.query(c_hi_lat))
    plt.annotate(text, xy=(1, 1), xycoords='axes fraction', ha='right', va='top',
                bbox=dict(boxstyle='round', alpha=0.3, facecolor='white'))


    # Adjust the layout for a better appearance
    plt.tight_layout()

    # Save the figure to a file (e.g., in PNG format)
    #plt.savefig('bayestar_extinction.png')

    # Show the plot
    plt.show()


#High extinction coordinate
    c_hi = SkyCoord(304.303, -32.6786, unit="deg")

test_data=pd.read_csv('/Users/mattocallaghan/XPNorm/Data/data_test_hi')[['parallax','ks_m' , 'phot_g_mean_mag', 'j_m', 'h_m']]
test_data=test_data[test_data['parallax']>0].reset_index(drop=True).values
for i in range(2,test_data.shape[1]):
    test_data[:,i]=test_data[:,i]-test_data[:,1]
test_data[:,1]=test_data[:,1]-2.5*np.log10(((1000/(test_data[:,0]))/10)**2)
test_data_scaled=(test_data-nf.mean)

logs = []
for i in np.arange(0, 0.3, 0.01):
    logs.append(nf.nfm.forward_kld(torch.tensor(test_data_scaled[:, 1:]) - extinction * i).mean().detach().numpy())

plt.plot(np.arange(0, 0.3, 0.01), logs)
plt.xlabel('E(B-V)')  # Add a label for the x-axis
plt.ylabel('P(E(B-V))')  # Add a label for the y-axis
plt.title('304.303, -32.6786')  # Add a title for the plot
text = 'Planck E(B-V)= '+str(sfd.query(c_hi))
plt.annotate(text, xy=(1, 1), xycoords='axes fraction', ha='right', va='top',
            bbox=dict(boxstyle='round', alpha=0.3, facecolor='white'))


# Adjust the layout for a better appearance
plt.tight_layout()

# Save the figure to a file (e.g., in PNG format)
plt.savefig('high_extinction.png')

# Show the plot
plt.show()
'''
    #Medium extinction


