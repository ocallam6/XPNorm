from cgi import test
from typing import final
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import csv
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

'''
This perpares the data for a normalising flow
'''

#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

from copyreg import pickle
import wave
import pandas as pd
from astropy.coordinates import SkyCoord, Galactic
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
#from gaiaxpy import convert,calibrate
import pickle
import re
from zero_point import zpt
import healpy as hp
import joblib

#############################################################################
###################### Data Generation Class ################################
#############################################################################
from tqdm.auto import tqdm

class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


class Data_Generation():
    def __init__(self):
        ebv_map = hp.read_map('/Users/mattocallaghan/XPNorm/Data/ebv_lhd.hpx.fits', verbose=False)
        #hp.mollview((ebv_map), title='', unit='log(E(B-V) [mag])')
        #hp.graticule()
        nside = hp.get_nside(ebv_map)
        npix = hp.nside2npix(nside)
        ordering = 'ring'
        pixel_indices = np.arange(npix)

        # Get the pixel centers
        l, b = hp.pix2ang(nside, pixel_indices,lonlat=True)


        idx=np.argwhere((~np.isnan(ebv_map))*(ebv_map<0.01))

        coords = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
        ra=coords.icrs.ra.degree
        dec=coords.icrs.dec.degree
        self.places=np.stack((ra[idx],dec[idx]),1)
        print(self.places.shape[0])
        #self.places=[[342.303, -47.6786],[343.9268,-51.2097],[345.6808,-51.7997],[342.3994,-49.7344],[340.8273,-49.0403]]
        #self.places=[[246.7065, 44.0438],[165.3404, 59.5480],[220.4940,59.9686],[160.4000,56.6931],[59.1394,-45.6599],[70.4262,-53.1910],[53.0764,-28.4748]
        #,[342.303, -47.6786],[343.9268,-51.2097],[345.6808,-51.7997],[342.3994,-49.7344],[340.8273,-49.0403]] #northern and southern
        #self.places=[[304.303, -59.6786]]
        #self.places=[[339.94503323, -24.53707374]] # vlow
        #self.places=[[309.50584297, -15.28053816]] #bayestar

        #self.places=[[293.79249997, 72.22446142]] #pasiphae slightly off centre of black
        #self.places=[[294.6134031, 72.33288884]] #pasiphae slightly off centre of black

        #self.places=[[295.31396796, 71.95873259]] #pasiphae red circle
        #self.places=[[294.46007047, 72.23165519]] #pasiphae black circle small

        #self.places=[[191.2318,57.1060]]
        #self.places=[[191.2318,45.1060]]

        #self.places=[[56.58101793, 23.9477124]] # merope
        
        self.PS=True
        self.bprp=False
        self.samples_loop=1
        self.separation=40.0/60
        
        #wavelength observations for BPRP
        self.numbers = np.arange(392,992,10)
        self.wls = [str(num) for num in self.numbers]

        self.bp_rp_data()

        self.data.to_csv('/Users/mattocallaghan/XPNorm/Data/data_full_ps_2')
        self.err.to_csv('/Users/mattocallaghan/XPNorm/Data/err_full_ps_2')



    def GetGAIAData_BPRP(self,arr):


        '''
        This takes an input of centres and creates 45' boxes with stars.
        '''

        x,y=arr   
        dfGaia = pd.DataFrame()
        if(self.bprp==True):
            #job = Gaia.launch_job_async( "select top 100 * from gaiadr2.gaia_source where parallax>0 and parallax_over_error>3. ") # Select `good' parallaxes
            qry = "SELECT * \
            FROM gaiadr3.gaia_source AS g, gaiaedr3_distance as d \
            WHERE DISTANCE(%f, %f, g.ra, g.dec) < 1\
            AND has_xp_continuous = 'True'\
            AND g.source_id = d.source_id;" % (x, y)
            job = Gaia.launch_job_async( qry )
            tblGaia = job.get_results()       #Astropy table
            dfGaia = tblGaia.to_pandas()      #convert to Pandas dataframe
        else:
                        #job = Gaia.launch_job_async( "select top 100 * from gaiadr2.gaia_source where parallax>0 and parallax_over_error>3. ") # Select `good' parallaxes
            qry = "SELECT * \
            FROM gaiadr3.gaia_source AS g, gaiaedr3_distance as d \
            WHERE DISTANCE(%f, %f, g.ra, g.dec) < 0.134\
            AND g.source_id = d.source_id;" % (x, y)
            job = Gaia.launch_job_async( qry )
            tblGaia = job.get_results()       #Astropy table
            dfGaia = tblGaia.to_pandas() 
        #print(len(dfGaia))
        
            
        return dfGaia

    def GetPSData(self,GaiaDR2SourceIDs):

        '''
        This cross matches the 2mass data based on a set of source ids from Gaia
        '''

        dfGaia = pd.DataFrame()
    
        qry ="SELECT * \
            FROM gaiadr3.gaia_source AS g \
            JOIN gaiaedr3.panstarrs1_best_neighbour AS xmatch USING (source_id) \
            JOIN gaiaedr3.panstarrs1_join AS xjoin USING (clean_panstarrs1_oid) \
            JOIN gaiadr2.panstarrs1_original_valid AS psdr1 ON xjoin.original_ext_source_id = psdr1.obj_id\
            WHERE  g.source_id in {}".format(GaiaDR2SourceIDs)

        
        job = Gaia.launch_job_async( qry )
        tblGaia = job.get_results()       #Astropy table
        dfGaia = tblGaia.to_pandas()  
        #print(len(dfGaia[dfGaia['number_of_neighbours'].astype(float)<2]))   
        #print(dfGaia.columns[-20:])
        return dfGaia[dfGaia['number_of_neighbours']<2].reset_index(drop=True)
    
    def Get2MASSData(self,GaiaDR2SourceIDs):

        '''
        This cross matches the 2mass data based on a set of source ids from Gaia
        '''

        dfGaia = pd.DataFrame()
    
        qry ="SELECT * \
            FROM gaiadr3.gaia_source AS g \
            JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id) \
            JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid) \
            JOIN gaiadr1.tmass_original_valid AS tmass ON xjoin.original_psc_source_id = tmass.designation \
            WHERE  g.source_id in {}".format(GaiaDR2SourceIDs)

        
        job = Gaia.launch_job_async( qry )
        tblGaia = job.get_results()       #Astropy table
        dfGaia = tblGaia.to_pandas()  
        #print(len(dfGaia))    #convert to Pandas dataframe
        return dfGaia[dfGaia['number_of_neighbours']<2].reset_index(drop=True)
    
    def create_gaia_data(self):


        # Function to fetch and write GAIA data
        def fetch_and_write_gaia_data(place, first_iteration):

            # Fetch GAIA data for the current place
            data = self.GetGAIAData_BPRP(place)
            
            # Convert the data to a DataFrame
            df = pd.DataFrame(data)
            
            if first_iteration:
                # For the first iteration, write the DataFrame with column names
                df.to_csv('/Users/mattocallaghan/XPNorm/Data/gaia_data_black.csv', index=False)
            else:
                # For subsequent iterations, append the DataFrame without column names
                df.to_csv('/Users/mattocallaghan/XPNorm/Data/gaia_data_black.csv', mode='a', header=False, index=False)

        # Define the number of processes (CPU cores) to use
        num_processes = 4  # Change this according to the number of CPU cores available

        # Create a list of arguments for each place
        args_list = [(place, i == 0) for i, place in enumerate(self.places)]

        # Use joblib to parallelize the task
        #with tqdm(total=len(self.places), desc="Fetching and Writing GAIA Data") as pbar:
        ProgressParallel(n_jobs=num_processes)(
                delayed(fetch_and_write_gaia_data)(args_list[i][0],args_list[i][1]) for i in range(len(args_list))
            )
            #pbar.update(len(self.places))


    def create_cross_match_data(self):
        data=pd.read_csv('/Users/mattocallaghan/XPNorm/Data/gaia_data_black.csv').drop_duplicates(subset=['source_id'])
        data=data[data['ruwe']<=1.4].reset_index(drop=True)
        data=data[data['phot_bp_mean_mag']<22].reset_index(drop=True)
        data=data[data['phot_rp_mean_mag']<22].reset_index(drop=True)
        data=data[data['phot_g_mean_mag']<19].reset_index(drop=True)
        data=data[data['phot_bp_mean_mag']<19].reset_index(drop=True) #check lallement et al
                #print(len(data))
        data['source_id']=data['SOURCE_ID']

        # 2MASS

        r2MASS=[]#self.Get2MASSData(tuple(data['source_id'].astype(int).to_numpy().astype(str)))
        chunks = np.array_split(data['source_id'].astype(int).astype(str), len(data) // 10000 + 1)

        for chunk in chunks:
            r2MASS.append(self.Get2MASSData(tuple(chunk)))
        r2MASS=pd.concat(r2MASS)
        r2MASS['source_id']=r2MASS['SOURCE_ID']
        combined_data=data.set_index('source_id').combine_first(r2MASS.set_index('source_id')).reset_index()
        combined_data=combined_data.dropna(subset='ks_m')
        # Make AAA selection cut on the 2MASS data
        flags=combined_data['ph_qual'].values
        flag_list=[]
        for i in range(len(flags)):
            if('X' in str(flags[i]) or ('U' in str(flags[i])) or ('F' in str(flags[i])) or ('E' in str(flags[i])) or ('D' in str(flags[i])) or ('C' in str(flags[i])) or ('B' in str(flags[i])) or ('n' in str(flags[i])) ):
                flag_list.append(False)
            else:
                flag_list.append(True)
        combined_data['accept_2mass']=np.array(flag_list)
        combined_data=combined_data[combined_data['accept_2mass']==True].reset_index(drop=True)
        combined_data=combined_data.drop_duplicates(subset='source_id')
        print(len(combined_data))
        # PS1
        chunks = np.array_split(combined_data['source_id'].astype(int).astype(str), len(data) // 10000 + 1)
        ps1=[]
        for chunk in chunks:
            ps1.append(self.GetPSData(tuple(chunk)))
        ps1=pd.concat(ps1)#self.GetPSData(tuple(combined_data['source_id'].astype(int).to_numpy().astype(str)))
        ps1['source_id']=ps1['SOURCE_ID']
        combined_data=combined_data.set_index('source_id').combine_first(ps1.set_index('source_id')).reset_index()
        combined_data=combined_data.drop_duplicates(subset='source_id')
        print(len(combined_data))

        return combined_data

    def bp_rp_data(self):
        # create the gaia pure data
        self.create_gaia_data()
        # cross match with 2MASS and PS

        merged_df=self.create_cross_match_data()
        
        if(self.bprp==True):
            idxs=split_range(len(merged_df),4999)
    
            for i in range(len(idxs)):
            #for i in range(1):
    
                y=calibrate(list(merged_df['source_id'].astype(str).values[idxs[i]]),sampling=np.arange(392,992,10))
                if(i==0):
                    bprp=y[0]
                else:
                    bprp=pd.concat([bprp,y[0]],axis=0)


            final_bprp = pd.merge(merged_df, bprp, how='inner', on='source_id')
        else:
            final_bprp=merged_df # bad naming convention - no bprp
        final_bprp = final_bprp.drop_duplicates(subset=['source_id'])
        final_bprp=final_bprp[final_bprp['ruwe']<=1.4].reset_index(drop=True)
        final_bprp=final_bprp[final_bprp['phot_bp_mean_mag']<22].reset_index(drop=True)
        final_bprp=final_bprp[final_bprp['phot_rp_mean_mag']<22].reset_index(drop=True)
        final_bprp=final_bprp[final_bprp['phot_g_mean_mag']<19].reset_index(drop=True)
        final_bprp=final_bprp[final_bprp['phot_bp_mean_mag']<19].reset_index(drop=True) #check lallement et al
 
       

       
        zpt.load_tables()
        final_bprp['zero_point']=final_bprp.apply(zpt.zpt_wrapper,axis=1)

        final_bprp['parallax']=final_bprp['parallax']-final_bprp['zero_point']
        if(self.bprp==True):
            for number in self.wls:
                column_name = number  # You can customize the column name
                final_bprp[column_name] = np.NaN
                final_bprp[column_name+'_err'] = np.NaN
            for i in range(len(final_bprp)):
                final_bprp.loc[i,self.wls]=-2.5*np.log10(np.abs(final_bprp['flux'][i]))
                final_bprp.loc[i,[self.wls[j]+'_err' for j in range(len(self.wls))]]=final_bprp['flux_error'][i]*(2.5/np.log(10))/np.abs(final_bprp['flux'][i])
    
            final_bprp['g_error']=np.sqrt(((1/final_bprp['phot_g_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            final_bprp['bp_error']=np.sqrt(((1/final_bprp['phot_bp_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            final_bprp['rp_error']=np.sqrt(((1/final_bprp['phot_rp_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            
            
            inputs=pd.concat([final_bprp[['ks_m','parallax','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m']],final_bprp[self.wls]],axis=1)
            error=pd.concat([final_bprp[['ks_msigcom','parallax_error','g_error','bp_error','rp_error','j_msigcom','h_msigcom']],final_bprp[[self.wls[i]+'_err' for i in range(len(self.wls))]]],axis=1)
            
        else:
    
            final_bprp['g_error']=np.sqrt(((1/final_bprp['phot_g_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            final_bprp['bp_error']=np.sqrt(((1/final_bprp['phot_bp_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            final_bprp['rp_error']=np.sqrt(((1/final_bprp['phot_rp_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            final_bprp['mu_error']=np.sqrt(((5/np.log(10))*(final_bprp['parallax_error']/final_bprp['parallax']))**2)
            final_bprp['mu']=2.5*np.log10((((1000)/final_bprp['parallax'])/10)**2)
            final_bprp=final_bprp[final_bprp['parallax']>0.0]
            final_bprp=final_bprp[(final_bprp['parallax_error']/final_bprp['parallax'])<0.2]

            final_bprp=final_bprp[final_bprp['ks_msigcom']<final_bprp['ks_msigcom'].mean()+3*final_bprp['ks_msigcom'].std()]
            final_bprp=final_bprp[final_bprp['ks_msigcom']<0.05]
            final_bprp=final_bprp[final_bprp['h_msigcom']<0.05]
            final_bprp=final_bprp[final_bprp['j_msigcom']<0.05]
            final_bprp=final_bprp[final_bprp['g_mean_psf_mag_error']<0.05]
            final_bprp=final_bprp[final_bprp['r_mean_psf_mag_error']<0.05]
            final_bprp=final_bprp[final_bprp['i_mean_psf_mag_error']<0.05]
            final_bprp=final_bprp[final_bprp['z_mean_psf_mag_error']<0.05]
            final_bprp=final_bprp[final_bprp['y_mean_psf_mag_error']<0.05]

            final_bprp=final_bprp[final_bprp['h_msigcom']<final_bprp['h_msigcom'].mean()+3*final_bprp['h_msigcom'].std()]
            final_bprp=final_bprp[final_bprp['j_msigcom']<final_bprp['j_msigcom'].mean()+3*final_bprp['j_msigcom'].std()]
            final_bprp=final_bprp[final_bprp['g_error']<final_bprp['g_error'].mean()+3*final_bprp['g_error'].std()]
            final_bprp=final_bprp[final_bprp['bp_error']<final_bprp['bp_error'].mean()+3*final_bprp['bp_error'].std()]
            final_bprp=final_bprp[final_bprp['rp_error']<final_bprp['rp_error'].mean()+3*final_bprp['rp_error'].std()].reset_index(drop=True)
            inputs=final_bprp[['ks_m','mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag', 'z_mean_psf_mag', 'y_mean_psf_mag','ra','dec']]
            error=final_bprp[['ks_msigcom','mu_error','g_error','bp_error','rp_error','j_msigcom','h_msigcom','g_mean_psf_mag_error','r_mean_psf_mag_error','i_mean_psf_mag_error','z_mean_psf_mag_error','y_mean_psf_mag_error','ra_error','dec_error']]
        for i in range(self.samples_loop):
            if(i==0):
                sigma=error.copy()
                x=inputs.copy()
            else:
                    
                x=pd.concat([x,inputs.copy()+np.random.normal(size=error.values.shape)*error.copy().values])
                sigma=pd.concat([sigma,error.copy()])

        x=x.reset_index(drop=True)
        sigma=sigma.reset_index(drop=True)
        #x=x.dropna().reset_index(drop=True)
        self.mean=x.values.mean(axis=0)[None,:]

        self.std=x.values.std(axis=0)[None,:]

        self.data=x

        self.err=sigma

def split_range(n, m):
    result = [list(range(i, min(i + m, n))) for i in range(0, n, m)]
    return result
Data_Generation()