import hmac
import matplotlib.pyplot as plt
from src.HMC_Single_Star_cosdist import HMC_Sampler
from src import NF_Cos_dist
import jax.numpy as jnp
import numpy as np
import matplotlib.colors as mcolors

if __name__=="__main__":
    #nf=NF_Cos_dist.JaxNormFlow()

    hmc=HMC_Sampler()

    #hmc.plot_profile()


    data_analysis=True
    if(data_analysis):
        train_data=hmc.normalising_flow.data+hmc.normalising_flow.mean
        train_distance=10**(((train_data[:,0])/(jnp.abs(jnp.sin(jnp.radians(hmc.normalising_flow.b))))+5)/5)
        

        test_data=hmc.data+hmc.mean
        test_distance=hmc.distance




        plt.figure()
        scatter = plt.scatter(
            train_data[:len(train_data)//32, 3] - train_data[:len(train_data)//32, 4],
            train_data[:len(train_data)//32, 7],
            s=1,
            c=2.5*np.log10((train_distance[:len(train_data)//32]/100)**2),
            cmap='viridis'
        )
        plt.colorbar(scatter, label='Radial distance')  # Add color bar if needed
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Radial distance HR diagram')
        plt.show()


        plt.figure()
        scatter = plt.scatter(
            train_data[:len(train_data)//32, 3] - train_data[:len(train_data)//32, 4],
            train_data[:len(train_data)//32, 7],
            s=1,
            c=train_data[:len(train_data)//32,0],
            cmap='viridis'
        )
        plt.colorbar(scatter, label='Perpendicular distance')  # Add color bar if needed
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()



    run_MCMC=False
    if(run_MCMC):

        error=[]
        ebvs=[]
        for i in range(len(hmc.data)):
            ebv,err=hmc.run_model_single(i)
            print(ebv)
            print(err)
            ebvs.append((ebv))
            error.append(err)

        #np.save('ebv_black',np.array(ebvs))
        #np.save('error_red',np.array(error))


    plot_distance=False
    if(plot_distance):
        mean=np.load('ebv_black.npy')/3.1
        stds=np.load('error_red.npy')/3.1
        dist=hmc.distance
        b=hmc.b/hmc.b.max()
        plt.errorbar(dist, mean, yerr=stds, fmt='o', ecolor='red', capsize=5, label='Mean with Error Bars')
        plt.xlabel('Distance')
        plt.ylabel('Mean Value')
        plt.title('Scatter Plot with Error Bars')
        plt.legend()
        plt.show()


