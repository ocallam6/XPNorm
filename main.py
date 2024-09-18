import hmac
import matplotlib.pyplot as plt
from src.HMC_Single_Star_cosdist import HMC_Sampler
from src import NF_Cos_dist
import jax.numpy as jnp
import numpy as np
import matplotlib.colors as mcolors

if __name__=="__main__":
    nf=NF_Cos_dist.JaxNormFlow()
    #nf.train()
    hmc=HMC_Sampler()

    #hmc.plot_profile()


    data_analysis=False
    if(data_analysis):
        train_data=hmc.normalising_flow.data+hmc.normalising_flow.mean
        train_distance=10**(((train_data[:,0])/(jnp.abs(jnp.sin(jnp.radians(hmc.normalising_flow.b))))+5)/5)
        

        test_data=hmc.data
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
            test_data[:, 3] - test_data[:, 4],
            test_data[:, 7],
            s=1,
            c=test_data[:,0],
            cmap='viridis'
        )
        plt.colorbar(scatter, label='Perpendicular distance')  # Add color bar if needed
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()






    def run_model_wrapper(index):
        x,y=hmc.run_model_single(index)
        
        return jnp.stack((x,y))

    import joblib
    from tqdm.auto import tqdm
    from joblib import Parallel, delayed
    from tqdm import tqdm

    class ProgressParallel(joblib.Parallel):
        def __call__(self, *args, **kwargs):
            with tqdm() as self._pbar:
                return joblib.Parallel.__call__(self, *args, **kwargs)

        def print_progress(self):
            self._pbar.total = self.n_dispatched_tasks
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()

    run_MCMC=False
    if(run_MCMC):
            print(hmc.data.shape)
            x=ProgressParallel(n_jobs=4)(
                delayed(run_model_wrapper)(i) for i in range(len(hmc.data))
            )
            x=jnp.stack(x,0)
            ebvs=x[:,0]
            error=x[:,1]
            np.save('ebv_black', np.array(ebvs))
            np.save('error_black', np.array(error))
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

        np.save('ebv_black',np.array(ebvs))
        np.save('error_black',np.array(error))


    plot_distance=False
    if(plot_distance):
        print(hmc.data.shape)

        #mean=np.load('ebv_black_no2.npy')/3.1   #these ones actually do have tmass....
        #stds=np.load('error_black_no2.npy')/3.1
        mean=np.load('ebv_highg.npy')/3.1  #these ones dont
        stds=np.load('error_highg.npy')/3.1
        x_err=hmc.mu_error
        print(x_err.shape)
        dist=hmc.distance
        print(dist.shape)
        dist_error=dist*(np.log(10)/5)*x_err

        idx=(mean<0.1)


        b=hmc.b/hmc.b.max()
        #plt.scatter(dist, mean, label='Mean with Error Bars')


#######################

        plt.figure()

        scatter2 = plt.scatter(
            train_data[:len(train_data)//32, 3] - train_data[:len(train_data)//32, 4],
            train_data[:len(train_data)//32, 7],s=1
        )
        scatter = plt.scatter(
            test_data[:, 3] - test_data[:, 4],
            test_data[:, 7],
            s=1,c=mean
        )
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()


        plt.figure()

        scatter2 = plt.scatter(
            train_data[:len(train_data)//32, -1],
            train_data[:len(train_data)//32, 0],s=1
        )
        scatter = plt.scatter(
            test_data[:, -1],
            test_data[:, 0],
            s=1,c=mean
        )
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()

        plt.figure()

        scatter2 = plt.scatter(
            train_data[:len(train_data)//32, -2],
            train_data[:len(train_data)//32, 0],s=1
        )
        scatter = plt.scatter(
            test_data[:, -2],
            test_data[:, 0],
            s=1,c=mean
        )
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()        
        plt.figure()

        scatter2 = plt.scatter(
            train_data[:len(train_data)//32, -3],
            train_data[:len(train_data)//32, 0],s=1
        )
        scatter = plt.scatter(
            test_data[:, -3],
            test_data[:, 0],
            s=1,c=mean
        )
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()        
        plt.figure()

        scatter2 = plt.scatter(
            train_data[:len(train_data)//32, -4],
            train_data[:len(train_data)//32, 0],s=1
        )
        scatter = plt.scatter(
            test_data[:, -4],
            test_data[:, 0],
            s=1,c=mean
        )
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()




        plt.figure()

        scatter2 = plt.scatter(
            train_data[:len(train_data)//32, -5],
            train_data[:len(train_data)//32, 0],s=1
        )
        scatter = plt.scatter(
            test_data[:, -5],
            test_data[:, 0],
            s=1,c=mean
        )
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()        
        
        plt.figure()

        scatter2 = plt.scatter(
            train_data[:len(train_data)//32, -6],
            train_data[:len(train_data)//32, 0],s=1
        )
        scatter = plt.scatter(
            test_data[:, -6],
            test_data[:, 0],
            s=1,c=mean
        )
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()        
        
        plt.figure()

        scatter2 = plt.scatter(
            train_data[:len(train_data)//32, -7],
            train_data[:len(train_data)//32, 0],s=1
        )
        scatter = plt.scatter(
            test_data[:, -7],
            test_data[:, 0],
            s=1,c=mean
        )
        plt.ylim(8, -2)
        plt.ylabel('$M_K$')
        plt.xlabel('$m_{BP}-m_{RP}$')
        plt.title('Perpendicular distance HR diagram')
        plt.show()



       #############        
 
        plt.errorbar(dist, mean, yerr=stds,xerr=dist_error, fmt='o', ecolor='red', capsize=5, label='Mean with Error Bars')
        plt.xlabel('Distance')
        plt.ylabel('Mean Value')
        plt.title('Scatter Plot with Error Bars')
        plt.legend()
        plt.show()

        plt.scatter(hmc.l[idx], hmc.b[idx],c=mean[idx])
        plt.colorbar()
        plt.legend()
        plt.show()






    LEARN_PROFILE=False
    if(LEARN_PROFILE):



        mean=jnp.load('ebv_black_no2.npy')/3.1
        stds=jnp.load('error_black_no2.npy')/3.1
        x_err=hmc.mu_error
        dist=jnp.array(hmc.distance)
        dist_error=dist*(np.log(10)/5)*x_err


        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d

        data=jnp.stack((dist,mean,stds),1)
        sorted_indices = np.argsort(data[:, 0])
        sorted_data = data[sorted_indices]

        # Sample data (replace with your actual data)
        x=sorted_data[:,0]
        y=sorted_data[:,1]
        y_err=sorted_data[:,2]
        def weighted_gaussian_filter1d(y, weights, sigma):
            # Normalize weights to sum to 1
            normalized_weights = weights / np.sum(weights)
            # Apply Gaussian filter to the weighted y values
            smoothed = gaussian_filter1d(y * normalized_weights, sigma=sigma)
            # Apply Gaussian filter to the weights
            weights_smoothed = gaussian_filter1d(normalized_weights, sigma=sigma)
            # Avoid division by zero
            weights_smoothed[weights_smoothed == 0] = 1e-10
            # Calculate weighted smoothed y values
            return smoothed / weights_smoothed

        # Apply the weighted Gaussian filter
        sigma = 0.1  # Adjust the sigma for smoothing level
        y_smooth_weighted = weighted_gaussian_filter1d(y, 1 / y_err**2, sigma)

# Ensure the smoothed data is non-decreasing
        y_non_decreasing = np.maximum.accumulate(y_smooth_weighted)


        # Ensure the smoothed data is non-decreasing
"""
        # Plot the results
        plt.errorbar(x, y, yerr=y_err, fmt='o', label='Original Data')
        plt.plot(x, y_smooth_weighted, label='Smoothed Data', linestyle='--')
        plt.plot(x, y_non_decreasing, label='Non-Decreasing Smoothed Data', linestyle='-')
        plt.xlabel('Distance (pc)')
        plt.ylabel('y values')
        plt.legend()
        plt.show()

        # Print the smoothed and non-decreasing values
        print("Smoothed Values:", y_smooth)
        print("Non-Decreasing Smoothed Values:", y_non_decreasing)






        data=jnp.einsum('ij,bj->bi', hmc.data_transform_inv,hmc.normalising_flow.data)
        data=jnp.einsum('ij,bj->bi', hmc.data_transform_2mass,data)
        #data=hmc.normalising_flow.data

        mu=(hmc.normalising_flow.cos_dist).mean(0)
        plt.scatter((hmc.normalising_flow.cos_dist)-mu,data[:,3]-data[:,4],s=1)
        scatter=plt.scatter((hmc.cos_dist)-mu,hmc.data[:,3]-hmc.data[:,4],c=mean,cmap='viridis')
        colorbar=plt.colorbar(scatter)
        plt.xlabel('Distance')
        plt.ylabel('I')
        colorbar.set_label('EBV')
        plt.show()


        plt.scatter(data[:,0],data[:,-1],s=1)
        scatter=plt.scatter(hmc.data[:,0],hmc.data[:,-1],c=dist,cmap='viridis')
        colorbar=plt.colorbar(scatter)
        plt.xlabel('BP-I')
        plt.ylabel('I')
        plt.ylim(6,-6)
        colorbar.set_label('Distance PC')
        plt.show()

        plt.scatter(data[:,3]-data[:,4],data[:,-1],s=1)
        scatter=plt.scatter(hmc.data[:,3]-hmc.data[:,4],hmc.data[:,-1],c=mean,cmap='viridis')
        colorbar=plt.colorbar(scatter)
        plt.xlabel('BP-I')
        plt.ylabel('I')
        plt.ylim(6,-6)
        colorbar.set_label('EBV')
        plt.show()


        plt.scatter(data[:,0],data[:,1],s=1)
        scatter=plt.scatter(hmc.data[:,0],hmc.data[:,1],c=mean,cmap='viridis')
        colorbar=plt.colorbar(scatter)
        plt.xlabel('Distance')
        plt.ylabel('I')
        colorbar.set_label('EBV')
        plt.show()"""