from src import Normalising_Flow_nodist
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nf=Normalising_Flow_nodist.Normalising_Flow_Trainer()
nf.load()


num_dim = 1
prior = utils.BoxUniform(low=-1 * torch.ones(num_dim), high=1 * torch.ones(num_dim))
coeffs= torch.tensor(np.array([
        [0.194059063720102, -0.000337880688254366, 0.000405004510990789, -0.000119030825664077, -2.90629429374213e-05, 9.85596051245887e-09, 1.22296149043372e-10, 0, 0, 0],
        [0.255058871064972, 7.19016588950228e-05, -0.000250455702483274, 3.99422163967702e-05, -6.83632867675118e-05, -2.3163568526324e-09, 7.26631781961228e-10, -2.27788077229475e-07, 4.97609167483581e-07, 6.67076034469308e-09],
        [0.340301468237771, -0.000826269158576803, -0.000283696380615497, 0.000103538996307887, -0.000156039957793959, 1.81587525109325e-07, 2.33928990111011e-09, 1.63733498839871e-06, 5.71693287820809e-08, 2.954302715354e-08],
        [0.663374149569189, -0.0184814428792349, 0.0100266536020118, -0.00317119320308867, -0.00649336268398495, 3.27674122186053e-05, 1.57894227641494e-06, -0.000116290469708794, 0.000117366662432525, 6.91273258513903e-06],
        [1.1516008149802, -0.0502982507379423, -0.00801054248601918, 0.0028487377407222, -0.0222045923218139, 0.000841943191161668, -1.31018008013547e-05, 0.00346423295251231, -0.000145621334026214, -6.85718568409361e-05],
        [0.993548688906439, -0.110149052160837, 0.0264447715065468, -0.00571010222810317, -0.0374363031107716, 0.00151447309438712, -2.52364537395156e-05, 0.00623824755961677, -0.000123598316318183, -0.000158499801004388]
        ]))




def f(x,a,i):
    
    coeff=coeffs[i]
    result=coeff[0]*torch.ones_like(x)
    result+=coeff[1]*x
    result+=coeff[2]*x**2
    result+=coeff[3]*x**3

    result+=coeff[4]*a
    result+=coeff[5]*a**2
    result+=coeff[6]*a**3

    result+=coeff[7]*x*a
    result+=coeff[8]*a*x**2
    result+=coeff[9]*x*a**2
    return result


def simulator(ebv):
    x=torch.tensor(nf.sample(num_samples=1))
    extinction_vector=torch.concatenate([f(x[:,-1:],ebv,int(i)) for i in np.arange(5,-1,-1)],axis=1)

    extinction_vector=torch.tensor(torch.einsum('ij,bj->bi',torch.tensor(nf.data_transform[1:,1:]),extinction_vector))

    return x[:,:]+torch.tensor(extinction_vector*ebv)

#posterior = infer(simulator, prior, method="SNPE", num_simulations=10000)


import pickle



#with open('filename.pickle', 'wb') as handle:
#    pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('posterior.pickle', 'rb') as handle:
#    posterior = pickle.load(handle)
with open('filename.pickle', 'rb') as handle:
    posterior = pickle.load(handle)

data_file='/Users/mattocallaghan/XPNorm/Data/data_full'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_test'

test_file='/Users/mattocallaghan/XPNorm/Data/data_noext'
test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle'

#test_file='/Users/mattocallaghan/XPNorm/Data/data_red_circle'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_black_circle_20'
#test_file='/Users/mattocallaghan/XPNorm/Data/data_noext'

err_file='/Users/mattocallaghan/XPNorm/Data/err'



data=pd.read_csv(data_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']].values

data_test=pd.read_csv(test_file)[['mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','ks_m']].values


data_transform=np.array([
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

extinction_vector=torch.tensor(np.einsum('ij,j->i',data_transform,
np.array([0,g.mean(), bp.mean(), rp.mean(), j.mean(), h.mean(), ks.mean()])))


data=np.einsum('ij,bj->bi',np.array(data_transform),np.array(data))

data=data[(data[:,1]<10)*(data[:,1]>-2)]
data=data[(data[:,0]<20)*(data[:,0]>2)]
data=data[:,1:]
mean=np.mean(data,axis=0)
std=np.std(data,axis=0)
data=(data-mean)

data_test=np.einsum('ij,bj->bi',np.array(data_transform),np.array(data_test))

data_test=data_test[(data_test[:,1]<10)*(data_test[:,1]>-2)]
data_test=data_test[(data_test[:,0]<20)*(data_test[:,0]>2)]
#dist_mean=data_test[:,0].mean()
#data_test[:,1:]=(data_test[:,1:]-mean)
##data_test[:,0]=data_test[:,0]-dist_mean

ext=[]
stds=[]
for i in range(len(data_test)//32):
    observation = data_test[i,1:]
    samples = posterior.sample((10000,), x=observation[None,:])
    ext.append(samples.mean())
    stds.append(samples.std())

distance=10**((data_test[:len(data_test)//32,0]+5)/5)
ebv_mean=np.array(ext)
ebv_std = np.sqrt(stds)

# Create scatter plot
#plt.errorbar(distance, ebv_mean/3.1, yerr=ebv_std/3.1, fmt='o', markersize=3, capsize=3)
plt.scatter(distance, ebv_mean/3.1)

plt.xlabel('Distance Mean')
plt.xlim(0,2000)
#plt.ylim(0,0.3)
plt.ylabel('EBV Mean')
plt.title('Scatter plot with error bars')
plt.grid(True)
plt.show()