"""
In this data we use the Kurucz data spectra to generate extinction coefficients
This will be learnt by a Neural network to be incorperated in our model later.
"""



from dustapprox.io import svo
import numpy as np
import pickle
from scipy.interpolate import CubicSpline
from scipy.interpolate import RegularGridInterpolator
import extinction as ext
from astropy.constants import R_sun,pc
from glob import glob
#############################################################################
###################### Filters #######################################
#############################################################################
# this is for the iscohrones
filters = ["Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3','2MASS_J','2MASS_H','2MASS_Ks','WISE_W1','WISE_W2','PS_g','PS_i','PS_r','PS_y','PS_z'] #add wise on later.
# this is for dustapprox
which_filters = ['GAIA/GAIA3.G','GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp','2MASS/2MASS.J', '2MASS/2MASS.H', '2MASS/2MASS.Ks','WISE/WISE.W1','WISE/WISE.W2','PAN-STARRS/PS1.g','PAN-STARRS/PS1.i','PAN-STARRS/PS1.r','PAN-STARRS/PS1.w','PAN-STARRS/PS1.y','PAN-STARRS/PS1.z']
passbands = svo.get_svo_passbands(which_filters)

#############################################################################
###################### Prepare stellar models #######################################
#############################################################################
def prepare_stellar_models(prefix,loc='/Users/mattocallaghan/VaeStar/Isochrones_data/Kurucz2003all/*.fl.dat.txt'):
    models = glob(loc)
    apfields = ['teff', 'logg', 'feh', 'alpha']
    wl=[]
    fehs=[]
    logg=[]
    teff=[]
    fluxes=[]

    label = 'teff={teff:4g} K, logg={logg:0.1g} dex, [Fe/H]={feh:0.1g} dex'
    for fname in models:
        data = svo.spectra_file_reader(fname)
        lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
        lamb = data['data']['WAVELENGTH'].values * lamb_unit
        truth=(data['data']['WAVELENGTH']<60000)*(data['data']['WAVELENGTH']>2000)
        data['data'][truth]
        if(data['alpha']['value']==0.0):
            wl.append(data['data'][truth]['WAVELENGTH'].values)#.reshape(-1, 2).mean(-1))
            fluxes.append(data['data'][truth]['FLUX'].values)#.reshape(-1, 2).mean(-1))
            fehs.append(data['feh']['value'])
            logg.append(data['logg']['value'])
            teff.append(data['teff']['value'])
        #flux=flux*curves[1](data['data']['WAVELENGTH'].values * lamb_unit,av,Rv=Rv)
    pars=np.stack([np.array(fehs),np.array(logg),np.array(teff)]).T
    feh_unique=np.unique(np.array(fehs))
    logg_unique=np.unique(np.array(logg))
    teff_unique=np.unique(np.array(teff))
    parameters=np.stack(np.meshgrid(feh_unique,logg_unique,teff_unique,indexing='ij'),axis=-1)
    flux_grid=(np.zeros((parameters.shape[:-1]+(len(wl[0]),)))*np.NaN)

    indices=[]
    for i,j,k in np.ndindex((parameters.shape[0:-1])):
        try:
            idx=np.where([(np.prod(pars[_]==parameters[i,j,k])) for _ in range(len(pars))])[0]
            
            flux_grid[i,j,k]=fluxes[int(idx)]
            indices.append(np.array([i,j,k]))
        except:
            continue

    interp=RegularGridInterpolator((feh_unique,logg_unique,teff_unique),flux_grid,method='linear',
                                    bounds_error=False, fill_value=np.NaN)   
    np.save(prefix+'/wavelength_ps1',wl[0])
    file = open(prefix+'/flux_interp_ps1', 'wb')
    pickle.dump(interp,file)
    file.close() 
# download svo synthetic spectra
def prepare_passbands(prefix):
    which_filters = ['GAIA/GAIA3.G','GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp','2MASS/2MASS.J', '2MASS/2MASS.H', '2MASS/2MASS.Ks','WISE/WISE.W1','WISE/WISE.W2','PAN-STARRS/PS1.g','PAN-STARRS/PS1.i','PAN-STARRS/PS1.r','PAN-STARRS/PS1.y','PAN-STARRS/PS1.z']
    passbands = svo.get_svo_passbands(which_filters)
    pb_interp=[]
    for pb in passbands:
        pb_interp.append(CubicSpline(np.array(pb.wavelength)*10,pb.transmit,extrapolate=False))
    file = open(prefix+'/passband_interp_ps1', 'wb')
    pickle.dump(pb_interp,file)
    file.close()
    
    
    passband_zeropoint_vega=np.array([pk.Vega_zero_flux.value for pk in passbands])
    np.save(prefix+'/zero_points_ps1',passband_zeropoint_vega)
#prepare_stellar_models(prefix='/Users/mattocallaghan/StarNEST/Data')
#prepare_passbands(prefix='/Users/mattocallaghan/StarNEST/Data')
#prepare_stellar_models('/Data')
# you need to run the above line to generate the necessary grids
# the following is what you would do to generate the spectra function

wavelength=np.load('/Users/mattocallaghan/StarNEST/Data/wavelength_ps1.npy')


def interpolators_call_ps1(prefix='/Users/mattocallaghan/StarNEST/Data'):
    file = open(prefix+'/flux_interp_ps1', 'rb')
    interp_flux=pickle.load(file)
    file.close() 

    file = open(prefix+'/passband_interp_ps1', 'rb')
    pb_interp=pickle.load(file)
    file.close()

    passband_zeropoint_vega=np.load(prefix+'/zero_points_ps1.npy')
    return interp_flux,pb_interp,passband_zeropoint_vega
    



interp_flux,pb_interp,passband_zeropoint_vega=interpolators_call_ps1()
spectra_interpolator=interp_flux
        #Transmission Functions
pb_interp=pb_interp
wavelength=wavelength
transmission=np.array([np.nan_to_num(pb_interp[passband](wavelength),nan=0.0) for passband in range(len(filters))])
d_denomenator=np.array([np.trapz(x=wavelength,y=wavelength*transmission[i]) for i in range(len(transmission))])
passband_zeropoint_vega=passband_zeropoint_vega

def flux_from_parameters(av,feh,logg,teff,Rv,R=1,d=10):
    if(np.array(d).shape==()):
        R,d,av=np.array(R).reshape(1),np.array(d).reshape(1),np.array(av).reshape(1)

    flux_values=np.nan_to_num(spectra_interpolator((feh,logg,teff)),nan=0.0)*(R[:,None]*R_sun)**2/(d[:,None]*pc)**2
    #extincted_flux=flux_values*(10**(-0.4*np.matmul(av[:,None],extinction.fitzpatrick99(self.wavelength,1.0,3.1,'aa')[None,:])))
    extincted_flux=np.array(flux_values*(np.exp(-1*np.matmul(av[:,None],ext.fitzpatrick99(wavelength,1.0,Rv,'aa')[None,:]))))

    numerator=(wavelength[None,:]*extincted_flux)[:,:,None]*transmission.T[None,:,:]
    out=(np.trapz(y=numerator,x=wavelength,axis=1)/d_denomenator)/passband_zeropoint_vega
    return -2.5*np.log10(out)


exts=[]
def extinction_calculation(av,feh,logg,teff,Rv):
    extinction=flux_from_parameters(av,feh,logg,teff,Rv,R=1,d=10)-flux_from_parameters(0,feh,logg,teff,Rv,R=1,d=10)
    return extinction

teff=np.arange(3500,10000,50)
logg=np.arange(0,5,0.05)
feh=np.arange(-3,0.5,0.1)
#Rv=np.arange(1,5,0.2)
av=np.arange(0.001,2,0.05)

mesh=np.meshgrid(teff,logg,feh,av)
teff=mesh[0].flatten()
logg=mesh[1].flatten()
feh=mesh[2].flatten()
av=mesh[3].flatten()
vals=[]
for i in range(len(teff)):
    if(i%1000):
        print(i/len(teff))
    exts.append(extinction_calculation(av[i],feh[i],logg[i],teff[i],3.1))
    vals.append(flux_from_parameters(0,feh[i],logg[i],teff[i],3.1))
    
np.save('exts_dense',np.concatenate(exts,0))
np.save('vals_dense',np.concatenate(vals,0))
    