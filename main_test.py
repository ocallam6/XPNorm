from cgi import test
from tokenize import Double
#from src import Normalising_Flow
from dustmaps.lenz2017 import Lenz2017Query
import dustmaps
from astropy.coordinates import SkyCoord,Galactic
from dustmaps.sfd import SFDQuery
import numpy as np
import pandas as pd
import torch
import seaborn as sn
import matplotlib.pyplot as plt
from dustmaps.bayestar import BayestarQuery
from functorch import jacfwd
from itertools import combinations
from src.Extinction_map import Extinction_Trainer,Extinction_Map,BNN_Extinction
#from src.Bayesian_NN_nodist import BayesianExtinction_Trainer
from src.Bayesian_NN_pos import BayesianExtinction_Trainer



from astropy.coordinates import SkyCoord
from dustmaps.planck import PlanckQuery

from dustmaps.bayestar import BayestarQuery
from dustmaps.lenz2017 import Lenz2017Query

from astropy.coordinates import SkyCoord
import astropy.units as u
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn


from src import Normalising_Flow_nodist,Normalising_Flow_Extinction
if __name__=="__main__":

    nf=Normalising_Flow_Extinction.Normalising_Flow_Trainer()
    print(nf.data.shape)
    nf.load()

    nf=BayesianExtinction_Trainer()
    #nf.train()
    nf.load()
    #nf.load()

   # Define Galactic coordinates (l, b) in degrees
    l = 104.2  # Galactic longitude in degrees
    b = 22.5   # Galactic latitude in degrees
    #b = 22.3   # Galactic latitude in degrees
    l,b=103.90, 21.97
    #red circle
    #l = 103.85  # Galactic longitude in degrees
    #b = 21.7  
    galactic_coord = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
    equatorial_coord = galactic_coord.transform_to('icrs')
    print(equatorial_coord)

    
# Initialize empty lists to store distance and EBV arrays
    all_distances = []
    all_ebv = []
    distance = 10 ** (((torch.tensor(nf.data_test[:len(nf.data_test)//5, 2:3])).detach().numpy()*nf.dist_std[-1] + nf.dist_mean[-1] + 5) / 5)
    l = torch.tensor(nf.data_test[:len(nf.data_test)//5, 0:1]).detach().numpy()*nf.dist_std[0] + nf.dist_mean[0]
    b = torch.tensor(nf.data_test[:len(nf.data_test)//5, 1:2]).detach().numpy()*nf.dist_std[1] + nf.dist_mean[1]
    


    idx=distance[:,0].argsort()
    #distance=np.arange(0.1,5000,1)[:,None]
    #distance_in=2.5*np.log10(distance**2/100)-nf.dist_mean
    # Run the loop 100 times and collect distance and EBV arrays
    for i in range(500):
        ebv = (nf.model(torch.tensor(nf.data_test[:len(nf.data_test)//5, 0:3])).detach().numpy())
        # this is an extinction assumption
        all_distances.append(distance)
        all_ebv.append(ebv/3.1)

        # Convert lists to numpy arrays
    # Convert arrays to numpy arrays
    all_distances = np.array(all_distances)
    all_ebv = np.array(all_ebv)

    # Calculate mean and standard deviation across all runs
    mean_distance = np.mean(all_distances, axis=0)
    mean_ebv = np.mean(all_ebv, axis=0)
    std_ebv = np.std(all_ebv, axis=0)

    

    # Create DataFrame
    data = pd.DataFrame(np.stack([b[:,0],l[:,0],(distance[:, 0]), mean_ebv[:, 0], std_ebv[:, 0]], axis=1),
                        columns=['b','l','Distance', 'MeanAv', 'STDAv'])
    data = data.sort_values(by='Distance', ascending=True)
    #data=data[data['Distance']>900]
    
    l_c=104.08
    b_c=22.31
    data1=data[(data['l']-l_c)**2+(data['b']-b_c)**2<=0.160**2]
    #data=data[(data['l']-l_c)**2+(data['b']-b_c)**2<0.16**2]
    l_c=103.90
    b_c=21.97
    data2=data[(data['l']-l_c)**2+(data['b']-b_c)**2<=0.160**2]
    

    # Create figure and axes for subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Plot the first subplot (original data)
    #sn.lineplot(data=data, x='Distance', y='MeanAv', label='Mean $A_V$', ax=axes[0])
    sn.scatterplot(data=data, x='l', y='b', ax=axes[0],s=10,hue='MeanAv',palette='magma')

    # Plot thick green vertical lines at each point
    points = [(346, 393), (1250, 2140)]
    #for point in points:
     #   axes[0].axvline(x=point[0], color='green', linewidth=1)
      #  axes[0].axvline(x=point[1], color='green', linewidth=1)

    # Calculate upper and lower bounds
    upper_bound = data['MeanAv'] + data['STDAv']
    lower_bound = data['MeanAv'] - data['STDAv']

    # Fill between upper and lower bounds
    #axes[0].fill_between(data['Distance'], lower_bound, upper_bound, alpha=0.3, label='$\pm \sigma$')
    #axes[0].set_xlim(0,4000)
    # Add labels and legend to the first subplot
    #axes[0].set_xlabel('Distance')
    #axes[0].set_ylabel('$E(B-V)$')
    #axes[0].legend()

    # Plot the second subplot (zoomed)
    #sn.lineplot(data=data, x='Distance', y='MeanAv', label='Mean $A_V$', ax=axes[1])
    sn.scatterplot(data=data1, x='Distance', y='MeanAv', ax=axes[1])
    sn.scatterplot(data=data2, x='Distance', y='MeanAv', ax=axes[1])
    for point in points:
        axes[1].axvline(x=point[0], color='green', linewidth=3)
        axes[1].axvline(x=point[1], color='green', linewidth=3)

    for point in points:
        axes[0].axvline(x=point[0], color='green', linewidth=3)
        axes[0].axvline(x=point[1], color='green', linewidth=3)

    # Plot thick green vertical lines at each point


    # Calculate upper and lower bounds for the zoomed plot
    upper_bound = data2['MeanAv'] + data2['STDAv']
    lower_bound = data2['MeanAv'] - data2['STDAv']

    # Fill between upper and lower bounds for the zoomed plot
    axes[1].fill_between(data2['Distance'], lower_bound, upper_bound, alpha=0.3, label='$\pm \sigma$')
    upper_bound = data1['MeanAv'] + data1['STDAv']
    lower_bound = data1['MeanAv'] - data1['STDAv']

    axes[0].fill_between(data1['Distance'], lower_bound, upper_bound, alpha=0.3, label='$\pm \sigma$')

    # Set the limits for the zoomed plot
    #axes[0].set_xlim(105, 103)
    axes[1].set_ylim(axes[1].get_ylim())

    # Add labels and legend to the second subplot
    axes[1].set_xlabel('Distance')
    axes[1].set_ylabel('$E(B-V)$')
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Show plot
    plt.show()
    # Show plot
    plt.show()


    

 
    '''
    planck = BayestarQuery()
    ebv = planck(galactic_coord,mode='samples')
    
    print(ebv)
    plt.plot(planck.distances*1000, ebv.mean(0)+ebv.std(0))
    plt.plot(planck.distances*1000, ebv.mean(0))
    plt.plot(planck.distances*1000, ebv.mean(0)-ebv.std(0))
    plt.xlim(0,4000)
    plt.show()    # Create a SkyCoord object with Galactic coordinates
'''

'''
    from astropy.coordinates import SkyCoord
    from dustmaps.planck import PlanckQuery

    from dustmaps.bayestar import BayestarQuery
    from dustmaps.lenz2017 import Lenz2017Query

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Define Galactic coordinates (l, b) in degrees
    l = 30.456  # Galactic longitude in degrees
    b = -30.345   # Galactic latitude in degrees

    # Create a SkyCoord object with Galactic coordinates
    galactic_coord = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
    equatorial_coord = galactic_coord.transform_to('icrs')
    print(equatorial_coord)
    planck = BayestarQuery(max_samples=1)
    ebv = planck(galactic_coord)
    print(ebv)
    plt.plot(planck.distances*1000, ebv)
    plt.xlim(0,4000)
    plt.show()
    data=pd.DataFrame(np.stack([z_masked[:,0],out.detach().numpy()[:,2],10*np.sqrt(10**((nf.data[:,1]+nf.mean[1])/2.5))],axis=1),columns=['truth','E(B-V)','distance'])
    
    plt.hist2d(data[data['truth']>0.5]['distance'],data[data['truth']>0.5]['E(B-V)'],bins=50)
    plt.title('Region with No Planck Extinction')
    plt.xlabel('Distance')
    plt.ylabel('E(B-V)')
    #sn.scatterplot(data[data['truth']>0.5],y='E(B-V)',x='distance')

    plt.show()

    '''