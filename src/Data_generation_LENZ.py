'''
This file generates data to be used by the Normalising Flow.
It cross matches Gaia (with and without BPRP spectra), 2MASS and PANSTARRS
It uses the LENZ dust map to look at regions at high Galactic latitude with 
low extinction to generate areas on the sky to query the Gaia database

'''


#############################################################################
###################### FILE LOCATIONS #######################################
#############################################################################

from math import comb
import wave
import pandas as pd
from astropy.coordinates import SkyCoord, Galactic
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
#from gaiaxpy import convert,calibrate

from zero_point import zpt
import healpy as hp
import joblib
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm import tqdm

#############################################################################
###################### Parallel Processing Task ################################
#############################################################################

class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

#############################################################################
###################### Data Generation Class ################################
#############################################################################

class Data_Generation():
    def __init__(self):
        """
        This class generates the data.
        1) Call the LENZ Map
            or select specific coordinates
        2) Choose which Bands should be included.
        3) Is the data for training the Normaliisng Flow or Inference
        4) Specify the angular separation in the search
        """

        self.places=self.Lenz_low_extinction_places()   
        self.places=[[339.94503323, -24.53707374]] # vlow
        #self.places=[[295.31396796, 71.95873259]] #pasiphae red circle
        #self.places=[[294.46007047, 72.23165519]] #pasiphae black circle small
        print(len(self.places))
        # What data will be include
        self.PS=True #PANSTARRS
        self.bprp=False #BPRP Spectra
        self.twomass_crossmatch=True
        # Is this for training the Normalising flow or inference
        self.TRAIN_FLOW=True
        if(self.TRAIN_FLOW):
            self.samples_loop=32
        else:
            self.samples_loop=1
        
        # What is the angular separation 
        self.separation=0.134 #LENZ 

        # CUTS ON DATA
        # GAIA
        self.ruwe_cut=1.4
        self.bp_bound_cut=19
        self.rp_bound_cut=22
        self.g_bound_cut=19
        self.error_over_parallax_cut=0.2

        # magnitude error upper bound
        self.mag_err_bound=0.05
        
        #wavelength observations for BPRP
        self.numbers = np.arange(392,992,10)
        self.wls = [str(num) for num in self.numbers]

        #Data is written to file
        #self.create_gaia_data(gaia_temp_store_location='/Users/mattocallaghan/XPNorm/Data/temp_gaia_data.csv')
        crossmatched_data=self.create_cross_match_data('/Users/mattocallaghan/XPNorm/Data/temp_gaia_data.csv')
        # the next stores the data in-class
        self.data,self.err=self.data_cuts(crossmatched_data)

        self.data.to_csv('/Users/mattocallaghan/XPNorm/Data/data_full_ps_2')
        self.err.to_csv('/Users/mattocallaghan/XPNorm/Data/err_full_ps_2')



    def Lenz_low_extinction_places(self):
        '''
        Query the LENZ Map and then return RA and DEC places
        '''
        # Call LENZ man
        ebv_map = hp.read_map('/Users/mattocallaghan/XPNorm/Data/ebv_lhd.hpx.fits', verbose=False)
        nside = hp.get_nside(ebv_map)
        npix = hp.nside2npix(nside)
        pixel_indices = np.arange(npix)
        # Get the pixel centers
        l, b = hp.pix2ang(nside, pixel_indices,lonlat=True)
        # Choose the indices where the EBV map is low.
        idx=np.argwhere((~np.isnan(ebv_map))*(ebv_map<0.008))
        #Query the sky coordinates.
        coords = SkyCoord(l=l*u.degree, b=b*u.degree, frame='galactic')
        ra=coords.icrs.ra.degree
        dec=coords.icrs.dec.degree
        return np.stack((ra[idx],dec[idx]),1)


    def create_gaia_data(self,gaia_temp_store_location):
        '''
        This function Creates the Gaia DataFrame
        If the BPRP is to be returned it also returns this.
        '''

        def fetch_and_write_gaia_data(place, first_iteration):

            # Fetch GAIA data for the current place
            data = self.GetGAIAData(place)
            
            # Convert the data to a DataFrame
            df = pd.DataFrame(data)
            
            if first_iteration:
                # For the first iteration, write the DataFrame with column names
                df.to_csv(gaia_temp_store_location, index=False)
            else:
                # For subsequent iterations, append the DataFrame without column names
                df.to_csv(gaia_temp_store_location, mode='a', header=False, index=False)

        # We run this data processing in parallel and iteratively append the csv
        # We set the number of processes to be the maximum number.
        num_processes = 4  
        args_list = [(place, i == 0) for i, place in enumerate(self.places)]
        ProgressParallel(n_jobs=num_processes)(
                delayed(fetch_and_write_gaia_data)(args_list[i][0],args_list[i][1]) for i in range(len(args_list))
            )
            #pbar.update(len(self.places))

    def GetGAIAData(self,arr):
        '''
        This takes an input of centres and creates circular region with stars within a 
        specified separation in the class definition.
        '''

        x,y=arr   
        dfGaia = pd.DataFrame()
        if(self.bprp==True):
            #job = Gaia.launch_job_async( "select top 100 * from gaiadr2.gaia_source where parallax>0 and parallax_over_error>3. ") # Select `good' parallaxes
            qry = "SELECT * \
            FROM gaiadr3.gaia_source AS g, gaiaedr3_distance as d \
            WHERE DISTANCE(%f, %f, g.ra, g.dec) < %f\
            AND has_xp_continuous = 'True'\
            AND g.source_id = d.source_id;" % (x, y,self.separation)
            job = Gaia.launch_job_async( qry )
            tblGaia = job.get_results()       #Astropy table
            dfGaia = tblGaia.to_pandas()      #convert to Pandas dataframe
        else:
                        #job = Gaia.launch_job_async( "select top 100 * from gaiadr2.gaia_source where parallax>0 and parallax_over_error>3. ") # Select `good' parallaxes
            qry = "SELECT * \
            FROM gaiadr3.gaia_source AS g, gaiaedr3_distance as d \
            WHERE DISTANCE(%f, %f, g.ra, g.dec) < %f\
            AND g.source_id = d.source_id;" % (x, y,self.separation)
            job = Gaia.launch_job_async( qry )
            tblGaia = job.get_results()       #Astropy table
            dfGaia = tblGaia.to_pandas() 
        
            
        return dfGaia

    def GetPSData(self,GaiaDR2SourceIDs):

        '''
        Given a list of IDs from Gaia this queries the Panstarrs database
        for the best cross match.
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

        #number_of_neighbours, which indicates how many sources 
        # in Pan-STARRS are matched with this source in Gaia.
        # Our cross match will pick only stars when there is one good match
        return dfGaia[dfGaia['number_of_neighbours']<2].reset_index(drop=True)
    
    def Get2MASSData(self,GaiaDR2SourceIDs):

        '''
        Given a list of IDs from Gaia this queries the 2MASS database
        for the best cross match.
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
        #number_of_neighbours, which indicates how many sources 
        # in 2mass are matched with this source in Gaia.
        # Our cross match will pick only stars when there is one good match
        return dfGaia[dfGaia['number_of_neighbours']<2].reset_index(drop=True)
    

    def create_cross_match_data(self,gaia_temp_store_location):
        # Call in the data and drop dupicates
        data=pd.read_csv(gaia_temp_store_location,low_memory=True).drop_duplicates(subset=['source_id'])

        # Make a RUWE_cut
        data=data[data['ruwe']<=self.ruwe_cut].reset_index(drop=True)
        # Make brightness cuts which have taken from Lallement et al
        # we make cuts here to avoid needlessly querying them from Gaia
        data=data[data['phot_bp_mean_mag']<self.bp_bound_cut].reset_index(drop=True)
        data=data[data['phot_rp_mean_mag']<self.rp_bound_cut].reset_index(drop=True)
        data=data[data['phot_g_mean_mag']<self.g_bound_cut].reset_index(drop=True)
        # this name change is to match 2MASS
        data['source_id']=data['SOURCE_ID']

        #############################################################################
        ###################### 2MASS ################################
        ##############################
        # We read the data in Chunks as the Gaia query has an upper limit on sources
        if(self.twomass_crossmatch):
            
            r2MASS=[]
            chunks = np.array_split(data['source_id'].astype(int).astype(str), len(data) // 10000 + 1)

            for chunk in chunks:
                r2MASS.append(self.Get2MASSData(tuple(chunk)))
            r2MASS=pd.concat(r2MASS)
            r2MASS['source_id']=r2MASS['SOURCE_ID']
            combined_data=data.set_index('source_id').combine_first(r2MASS.set_index('source_id')).reset_index()
            # Drop the data based on assuming everything must have a KS magnitude
            combined_data=combined_data.dropna(subset='ks_m')
            # Make AAA selection cut on the 2MASS data
            flags=combined_data['ph_qual'].values
            flag_list=[]

            #MAKE a cut on the flags. Has to be A quality
            for i in range(len(flags)):
                if('X' in str(flags[i]) or ('U' in str(flags[i])) or ('F' in str(flags[i])) or ('E' in str(flags[i])) or ('D' in str(flags[i])) or ('C' in str(flags[i])) or ('B' in str(flags[i])) or ('n' in str(flags[i])) ):
                    flag_list.append(False)
                else:
                    flag_list.append(True)
            combined_data['accept_2mass']=np.array(flag_list)
            combined_data=combined_data[combined_data['accept_2mass']==True].reset_index(drop=True)
            combined_data=combined_data.drop_duplicates(subset='source_id')
        else:
            combined_data=data
        #############################################################################
        ###################### PANSTARRS ################################
        ##############################

        if(self.PS):
            chunks = np.array_split(combined_data['source_id'].astype(int).astype(str), len(data) // 10000 + 1)
            ps1=[]
            for chunk in chunks:
                ps1.append(self.GetPSData(tuple(chunk)))
            ps1=pd.concat(ps1)#self.GetPSData(tuple(combined_data['source_id'].astype(int).to_numpy().astype(str)))
            ps1['source_id']=ps1['SOURCE_ID']
            combined_data=combined_data.set_index('source_id').combine_first(ps1.set_index('source_id')).reset_index()
            combined_data=combined_data.drop_duplicates(subset='source_id')

        # Rerturn the cross matched data
        return combined_data

    def data_cuts(self,merged_df):
        """
        This makes all extra cuts on the data and stores the data in the class
        """
        
        if(self.bprp==True):
            raise NotImplementedError
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
        final_bprp=final_bprp[final_bprp['ruwe']<=self.ruwe_cut].reset_index(drop=True)
        final_bprp=final_bprp[final_bprp['phot_rp_mean_mag']<self.rp_bound_cut].reset_index(drop=True)
        final_bprp=final_bprp[final_bprp['phot_g_mean_mag']<self.g_bound_cut].reset_index(drop=True)
        final_bprp=final_bprp[final_bprp['phot_bp_mean_mag']<self.bp_bound_cut].reset_index(drop=True) #check lallement et al
 
       

        ########
        ## Load the zero points of the parallax
        ########
        zpt.load_tables()
        final_bprp['zero_point']=final_bprp.apply(zpt.zpt_wrapper,axis=1)
        final_bprp['parallax']=final_bprp['parallax']-final_bprp['zero_point']
        
        if(self.bprp==True):
            raise NotImplementedError
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
            ####
            #### Define the Gaia magnitude errors
            ####
            final_bprp['g_error']=np.sqrt(((1/final_bprp['phot_g_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            final_bprp['bp_error']=np.sqrt(((1/final_bprp['phot_bp_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            final_bprp['rp_error']=np.sqrt(((1/final_bprp['phot_rp_mean_flux_over_error']*(2.5/np.log(10))))**2+0.003**2)
            #### Define the Gaia distance mu and tis correpsonding error
            final_bprp['mu_error']=np.sqrt(((5/np.log(10))*(final_bprp['parallax_error']/final_bprp['parallax']))**2)
            final_bprp['mu']=2.5*np.log10((((1000)/final_bprp['parallax'])/10)**2)
            
            if(True):
            #if(self.TRAIN_FLOW):
                """
                If the model is being trained for the cut the parallax to be positive
                Currently: always greater than zero
                """
                final_bprp=final_bprp[final_bprp['parallax']>0.0]
            
            ####
            #### Parallax error cuts, the cut needs to be made in the training loop
            #### to ensure we can reliably treat MU as Gaussian
            ####
            final_bprp=final_bprp[(final_bprp['parallax_error']/final_bprp['parallax'])<self.error_over_parallax_cut]

            ####
            #### Parallax error cuts, the cut needs to be made in the training loop
            ####

            final_bprp=final_bprp[final_bprp['ks_msigcom']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['h_msigcom']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['j_msigcom']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['g_mean_psf_mag_error']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['r_mean_psf_mag_error']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['i_mean_psf_mag_error']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['z_mean_psf_mag_error']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['y_mean_psf_mag_error']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['g_error']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['bp_error']<self.mag_err_bound]
            final_bprp=final_bprp[final_bprp['rp_error']<self.mag_err_bound].reset_index(drop=True)
            # other cuts may need to be made.
            
            inputs=final_bprp#[['ks_m','mu','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','j_m','h_m','g_mean_psf_mag', 'r_mean_psf_mag', 'i_mean_psf_mag', 'z_mean_psf_mag', 'y_mean_psf_mag','ra','dec']]
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

        return x,sigma

def split_range(n, m):
    result = [list(range(i, min(i + m, n))) for i in range(0, n, m)]
    return result
Data_Generation()