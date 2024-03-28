from cgi import test
from tokenize import Double
from dustmaps.lenz2017 import Lenz2017Query
import dustmaps
from astropy.coordinates import SkyCoord,Galactic
from dustmaps.sfd import SFDQuery
import numpy as np
import pandas as pd
import torch
import seaborn as sn
import matplotlib.pyplot as plt

from itertools import combinations


from src.Normalising_Flow_nodist import Normalising_Flow_Trainer



from src import Normalising_Flow_nodist
if __name__=="__main__":

    nf=Normalising_Flow_Trainer()
    nf.load()

    x=[nf.sample(len(nf.data)//32) for _ in range(10)]
    x=np.concatenate(x)
    plt.figure(figsize=(15, 7))  # Adjust the figure size as needed for your paper layout
    fig,axes=plt.subplots(1,2)
    fig.set_figheight(4*2)
    fig.set_figwidth(8*2)
    #axes[0].hist2d(nf.data[:,3-1]*nf.std[3-1]+nf.mean[3-1], nf.data[:,1-1]*nf.std[0]+nf.mean[1-1],bins=500)  # Adjust the colormap as needed
    axes[0].scatter(nf.data[:,3-1]+nf.mean[3-1], nf.data[:,1-1]*nf.std[0]+nf.mean[1-1])  # Adjust the colormap as needed

    # Creating the 2D histogram
    #axes[1].hist2d(x[:,3-1], x[:,1-1],bins=500)  # Adjust the colormap as needed
    axes[1].scatter(x[:,3-1], x[:,1-1])  # Adjust the colormap as needed

    axes[0].set_xlabel('$m_{BP}-m_{G}$', fontsize=12)  # Adjust the label and font size as needed
    axes[0].set_ylabel('$M_{G}$', fontsize=12)  
    # Setting labels and title
    axes[1].set_xlabel('$m_{BP}-m_{G} \sim p(x)$', fontsize=12)  # Adjust the label and font size as needed
    axes[1].set_ylabel('$M_{G} \sim p(x)$', fontsize=12)  # Adjust the label and font size as needed
    #plt.title('2D Histogram', fontsize=14)  # Adjust the title and font size as needed

    # Setting axis limits
    axes[0].set_ylim(10, 0)  # Reverse the y-axis to have increasing values from bottom to top
    axes[0].set_xlim(0,1.5)
    axes[1].set_ylim(10, 0)  # Reverse the y-axis to have increasing values from bottom to top
    axes[1].set_xlim(0,1.5)
    # Adding a colorbar for better interpretation of the histogram
    #cbar = plt.colorbar()
    #cbar.set_label('N Stars', fontsize=12)  # Adjust the label and font size as needed
    plt.tight_layout()
    # Saving the plot
    #plt.savefig('/Users/mattocallaghan/XPNorm/Plots/hr_d_sampled.png', bbox_inches='tight')  # Adjust filename and DPI as needed
    #plt.savefig('/Users/mattocallaghan/XPNorm/Plots/hr_d_sampled.pdf', bbox_inches='tight')  # Adjust filename and DPI as needed

    plt.show()


    fig,axes=plt.subplots(2,3)
    axes=axes.flatten()
    fig.set_figheight(6)
    fig.set_figwidth(10)
    sn.kdeplot((nf.data[:,0]+nf.mean[0]),label='Data',ax=axes[0],fill=True)  # Adjust the colormap as needed
    sn.kdeplot(x[:,0],label='Flow',ax=axes[0],fill=True,alpha=0.3)

    sn.kdeplot(nf.data[:,1]+nf.mean[1],label='Data',ax=axes[1],fill=True)  # Adjust the colormap as needed
    sn.kdeplot(x[:,1],label='Flow',ax=axes[1],fill=True,alpha=0.3)

    sn.kdeplot(nf.data[:,2]+nf.mean[2],label='Data',ax=axes[2],fill=True)  # Adjust the colormap as needed
    sn.kdeplot(x[:,2],label='Flow',ax=axes[2],fill=True,alpha=0.3)

    sn.kdeplot(nf.data[:,3]+nf.mean[3],label='Data',ax=axes[3],fill=True)  # Adjust the colormap as needed
    sn.kdeplot(x[:,3],label='Flow',ax=axes[3],fill=True,alpha=0.3)

    sn.kdeplot(nf.data[:,4]+nf.mean[4],label='Data',ax=axes[4],fill=True)  # Adjust the colormap as needed
    sn.kdeplot(x[:,4],label='Flow',ax=axes[4],fill=True,alpha=0.3)

    sn.kdeplot(nf.data[:,5]+nf.mean[5],label='Data',ax=axes[5],fill=True)  # Adjust the colormap as needed
    sn.kdeplot(x[:,5],label='Flow',ax=axes[5],fill=True,alpha=0.3)

    #sn.kdeplot(nf.data[:,6]+nf.mean[6],label='Data',ax=axes[6],fill=True)  # Adjust the colormap as needed
    #sn.kdeplot(x[:,6],label='Flow',ax=axes[6],fill=True,alpha=0.3)

    #sn.kdeplot(nf.data[:,3]+nf.mean[3]-(nf.data[:,2]+nf.mean[2]),label='Data',ax=axes[7],fill=True)  # Adjust the colormap as needed
    #sn.kdeplot(x[:,3]-x[:,2],label='Flow',ax=axes[7],fill=True,alpha=0.3)

    #sn.kdeplot(nf.data[:,-1]+nf.mean[-1]+(nf.data[:,1]+nf.mean[1]),label='Data',ax=axes[8],fill=True)  # Adjust the colormap as needed
    #sn.kdeplot(x[:,-1]+x[:,1],label='Flow',ax=axes[8],fill=True,alpha=0.3)

    #axes[0].set_title('$\mu$', fontsize=12)  # Adjust the label and font size as needed
    axes[0].set_title('$m_G-\mu$', fontsize=12)  # Adjust the label and font size as needed
    axes[1].set_title('$m_{BP}-m_{G}$', fontsize=12)  # Adjust the label and font size as needed
    axes[2].set_title('$m_{RP}-m_{G}$', fontsize=12)  # Adjust the label and font size as needed
    axes[3].set_title('$m_{J}-m_{G}$', fontsize=12)  # Adjust the label and font size as needed
    axes[4].set_title('$m_{H}-m_{G}$', fontsize=12)  # Adjust the label and font size as needed
    axes[5].set_title('$m_{K_s}-m_{G}$', fontsize=12)  # Adjust the label and font size as needed
    #axes[7].set_title('$m_{BP}-m_{RP}$', fontsize=12)  # Adjust the label and font size as needed
    #axes[8].set_title('$m_{K}-\mu$', fontsize=12)  # Adjust the label and font size as needed

    plt.legend()
    # Adding a colorbar for better interpretation of the histogram
    #cbar = plt.colorbar()
    #cbar.set_label('N Stars', fontsize=12)  # Adjust the label and font size as needed
    plt.tight_layout()
    # Saving the plot
    #plt.savefig('/Users/mattocallaghan/XPNorm/Plots/par_compare.png', bbox_inches='tight')  # Adjust filename and DPI as needed
    #plt.savefig('/Users/mattocallaghan/XPNorm/Plots/par_compare.pdf', bbox_inches='tight')  # Adjust filename and DPI as needed

    plt.show()