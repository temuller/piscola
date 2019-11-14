import piscola
from .filter_integration import *
from .gaussian_process import *
from .spline import *
from .extinction_correction import *
from .mangling import *  

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from peakutils import peak
import pandas as pd
import numpy as np
import pickle
import math
import glob
import os

### Initialisation functions ###

def initialise(file_name):
    """Initialise the 'sn' object.
    
    The object is initialise with all the necessary information like filters, fluxes, etc.
    
    Parameters
    ----------
    file_name : str
        Name of the SN or SN file.
        
    Returns
    -------
    New 'sn' object.
    
    """
    
    name, z, ra, dec = pd.read_csv(file_name, delim_whitespace=True, nrows=1).iloc[0].values
    sn_file = pd.read_csv(file_name, delim_whitespace=True, skiprows=2)
    sn_file.columns = sn_file.columns.str.lower()
    
    # call sn object
    sn_obj = sn(name, z=z, ra=ra, dec=dec)
    sn_obj.bands = [band for band in list(sn_file['band'].unique()) if len(sn_file[sn_file['band']==band]['flux']) >= 3]
    sn_obj.call_filters()
    
    # order filters by wavelength
    eff_waves = [sn_obj.filters[band]['eff_wave'] for band in sn_obj.bands]
    sorted_idx = sorted(range(len(eff_waves)), key=lambda k: eff_waves[k])
    sn_obj.bands = [sn_obj.bands[x] for x in sorted_idx]
    
    # add data of every band
    for band in sn_obj.bands:
        band_info = sn_file[sn_file['band']==band]
        sn_obj.data[band] = {'mjd':band_info['mjd'].values, 
                             'flux':band_info['flux'].values, 
                             'flux_err':band_info['flux_err'].values, 
                             'zp':float(band_info['zp'].unique()[0]),
                             'mag_sys':band_info['mag_sys'].unique()[0],
                            }
    
    sn_obj.set_sed_template()
    
    return sn_obj
    
    
def sn_file(str, directory='data/'):
    """Loads a supernova from a file.
    
    Parameters
    ----------
    directory : str, default 'data/'
        Directory where to look for the SN data files.
    
    """

    if os.path.isfile(directory+str):
        return initialise(directory+str)
    
    elif os.path.isfile(directory+str+'.dat'):    
        return initialise(directory+str+'.dat')
    
    elif os.path.isfile(str):
        return initialise(str)    
    
    else:
        raise ValueError(f'{str} was not a valid SN name or file.')

def load_sn(name, path=None):
    """Loads a 'sn' oject that was previously saved as a pickle file.
    
    Parameters
    ----------
    name : str
        Name of the SN object.
        
    Returns
    -------
    'sn' object previously saved as a pickle file.
    
    """
    
    if path is None:
        with open(name + '.pisco', 'rb') as file:
            return pickle.load(file)
    else:
        with open(path + name + '.pisco', 'rb') as file:
            return pickle.load(file)

#################################

class sn(object):
    """Supernova class for representing a supernova."""
    
    def __init__(self, name, z=0, ra=None, dec=None):
        self.name = name
        self.z = z # redshift
        self.ra = ra # coordinates in degrees
        self.dec = dec
        
        if self.ra is None or self.dec is None:
            print('Warning, ra and/or dec not specified')
            self.ra , self.dec = 0, 0
         
        self.__dict__['data'] = {}  # data for each band
        self.__dict__['sed'] = {}  # sed info
        self.__dict__['filters'] = {}  # filter info for each band
        self.__dict__['lc_fits'] = {}  # gp fitted data
        self.__dict__['lc_interp'] = {}  # interpolated data to be used for light curves correction
        self.__dict__['lc_correct'] = {}  # corrected light curves
        self.__dict__['lc_final_fits'] = {}  # interpolated corrected light curves 
        self.__dict__['lc_parameters'] = {}  # final SN light-curves parameters
        self.__dict__['sed_results'] = {}  # final SED for every phase if successful
        self.__dict__['mangling_results'] = {}  # mangling results for every phase if successful
        self.bands = None
        self.pivot_band = None
        self.tmax = None
        self.phase = 0 # Initial value for approximate effefctive wavelength calculation
        self.normalization = None  # NOT USED ANYMORE(?)
        self.test = None  # to test stuff - not part of the release
    
        
    def __repr__(self):
        return f'name = {self.name}, z = {self.z:.5}, ra = {self.ra}, dec = {self.dec}'
    
    def __getattr__(self, attribute):
        if attribute=='name':
            return self.name
        if attribute=='z':
            return self.z
        if attribute=='ra':
            return self.ra
        if attribute=='dec':
            return self.dec
        if 'data' in self.__dict__:
            if attribute in self.data:
                return(self.data[attribute])
        else:
            return f'Attribute {attribute} is not defined.'
        
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self, d):
        self.__dict__.update(d)
        
    def save(self, name=None, path=None):
        '''Saves a SN object into a pickle file'''
        
        if name is None:
            name = self.name
            
        if path is None:
            with open(name + '.pisco', 'wb') as pfile:
                pickle.dump(self, pfile, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path + name + '.pisco', 'wb') as pfile:
                pickle.dump(self, pfile, pickle.HIGHEST_PROTOCOL)

        
    #######################################################
    ###################### filters ########################
    #######################################################
    
    def call_filters(self):  
        """Obtains the filters's transmission function for the observed bands and the Bessell bands."""
        
        path = piscola.__path__[0]
        vega_wave, vega_flux = np.loadtxt(path + '/templates/alpha_lyr_stis_005.ascii').T
                    
        # add filters of the observed bands
        for band in self.bands:
            file = f'{band}.dat'

            for root, dirs, files in os.walk(path + '/filters/'):
                if file in files:
                    wave0, transmission0 = np.loadtxt(os.path.join(root, file)).T
                    # linearly interpolate filters
                    wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
                    transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                    # to anchor the transmission function on the edges to 0.0
                    mask = np.nonzero(transmission)[0]  # could have negative transmission(?)
                    
                    if mask[-1]+1 in np.where(transmission>-99)[0]:
                        mask = np.r_[mask, mask[-1]+1]
                    if mask[0]-1 in np.where(transmission>-99)[0]:
                        mask = np.r_[mask[0]-1, mask]
                        
                    wave, transmission = wave[mask], transmission[mask]
                    response_type = 'photon'
                    self.filters[band] = {'wave':wave, 
                                          'transmission':transmission, 
                                          'eff_wave':calc_eff_wave(vega_wave, vega_flux, wave, 
                                                                   transmission, response_type=response_type),
                                          'pivot_wave':calc_pivot_wave(wave, transmission, 
                                                                       response_type=response_type),
                                          'response_type':response_type}
        
        # add Bessell filters        
        file_paths = [file for file in glob.glob(path + '/filters/Bessell/*.dat')]

        for file_path in file_paths:
            band = os.path.basename(file_path).split('.')[0]
            wave0, transmission0 = np.loadtxt(file_path).T
            # linearly interpolate filters
            wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
            transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
            # to anchor the transmission function on the edges to 0.0
            mask = np.nonzero(transmission)[0]  # could have negative transmission(?)
            
            if mask[-1]+1 in np.where(transmission>-99)[0]:
                mask = np.r_[mask, mask[-1]+1]
            if mask[0]-1 in np.where(transmission>-99)[0]:
                mask = np.r_[mask[0]-1, mask]
                
            wave, transmission = wave[mask], transmission[mask]
            response_type = 'energy'
            self.filters[band] = {'wave':wave, 
                                  'transmission':transmission, 
                                  'eff_wave':calc_eff_wave(vega_wave, vega_flux, wave, transmission, 
                                                           response_type=response_type),
                                  'pivot_wave':calc_pivot_wave(wave, transmission, response_type=response_type),
                                  'response_type':response_type}
        
        
    def add_filters(self, filter_list, response_type='photon'):
        """Add choosen filters. You can add a complete directory with filters in it or add filters given in a list.
        
        Parameters
        ----------
        filter_list : list
            List of bands.
        response_type : str, default 'photon'
            Response type of the filter. The only options are: 'photon' and 'energy'.
            Only the Bessell filters use energy response type.

        """
        
        path = piscola.__path__[0]
        vega_wave, vega_flux = np.loadtxt(path + '/templates/alpha_lyr_stis_005.ascii').T
        
        if isinstance(filter_list, str) and os.path.isdir(f'{path}/filters/{filter_list}'):
            # add directory
            path = piscola.__path__[0]
            path = f'{path}/filters/{filter_list}'
            for file in os.listdir(path):
                if file[-4:]=='.dat':
                    band = file.split('.')[0]
                    wave0, transmission0 = np.loadtxt(os.path.join(path, file)).T
                    # linearly interpolate filters
                    wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
                    transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                    # to anchor the transmission function on the edges to 0.0
                    mask = np.nonzero(transmission)[0]  # could have negative transmission(?)
                    
                    if mask[-1]+1 in np.where(transmission>-99)[0]:
                        mask = np.r_[mask, mask[-1]+1]
                    if mask[0]-1 in np.where(transmission>-99)[0]:
                        mask = np.r_[mask[0]-1, mask]
                        
                    wave, transmission = wave[mask], transmission[mask]
                    self.filters[band] = {'wave':wave, 
                                          'transmission':transmission, 
                                          'eff_wave':calc_eff_wave(vega_wave, vega_flux, wave, transmission, 
                                                                   response_type=response_type), 
                                          'pivot_wave':calc_pivot_wave(wave, transmission, 
                                                                       response_type=response_type),
                                          'response_type':response_type}
                    
        else:   
            # add filters in list
            for band in filter_list:
                file = f'{band}.dat'
                
                for root, dirs, files in os.walk(path + '/filters/'):
                    if file in files:
                        wave, transmission = np.loadtxt(os.path.join(root, file)).T
                        # to anchor the transmission function on the edges to 0.0
                        mask = np.nonzero(transmission)[0]  # could have negative transmission(?)
                        
                        if mask[-1]+1 in np.where(transmission>-99)[0]:
                            mask = np.r_[mask, mask[-1]+1]
                        if mask[0]-1 in np.where(transmission>-99)[0]:
                            mask = np.r_[mask[0]-1, mask]
                            
                        wave, transmission = wave[mask], transmission[mask]
                        self.filters[band] = {'wave':wave, 
                                              'transmission':transmission, 
                                              'eff_wave':calc_eff_wave(ab_wave, ab_flux, wave, transmission, 
                                                                       response_type=response_type),
                                              'pivot_wave':calc_pivot_wave(wave, transmission, 
                                                                           response_type=response_type),
                                              'response_type':response_type}
            
            
    def plot_filters(self, filter_list=None, save=False):
        """Plot the filters' transmission functions.
        
        Parameters
        ----------
        filter_list : list, default 'None'
            List of bands.
        save : bool, default 'False'
            If true, saves the plot into a file.

        """
        
        if filter_list is None:
            filter_list = self.bands
        
        f, ax = plt.subplots(figsize=(8,6))    
        for band in filter_list:      
            norm = self.filters[band]['transmission'].max()
            ax.plot(self.filters[band]['wave'], self.filters[band]['transmission']/norm, label=band)
            
        ax.set_xlabel(r'wavelength ($\AA$)', fontsize=18, family='serif')
        ax.set_ylabel('normalized response', fontsize=18, family='serif')
        ax.set_title(r'Filters response functions', fontsize=18, family='serif')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.minorticks_on()
        ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        if save:
            f.tight_layout()
            plt.savefig('plots/filters.png')
            
        plt.show()
        
        
    def calc_pivot(self, band_list=None):
        """Calculates the observed band closest to Bessell-B band.
        
        The pivot band will be the band on which the SED phases will be
        based on during the first cycle of fits.
        
        Parameters
        ----------
        filter_list : list, default 'None'
            List of bands.

        """
        
        BessellB_eff_wave = self.filters['Bessell_B']['eff_wave']  # effective wavelength in Angstroms
        
        if band_list is None:
            band_list = self.bands
        
        bands_eff_wave =  np.asarray([self.filters[band]['eff_wave']/(1+self.z) for band in band_list])      
        idx = (np.abs(BessellB_eff_wave - bands_eff_wave)).argmin()  
        self.pivot_band = band_list[idx]
            
        
    #######################################################
    ######################## SED ##########################
    #######################################################
    
    def print_sed_templates(self):
        """Prints all the available SED templates in the 'templates' directory"""
        
        path = piscola.__path__[0]
        print('The list of available SED templates are:', [name for name in os.listdir(path + "/templates/") 
                                                           if os.path.isdir(f"{path}/templates/{name}")])
    
    
    def set_sed_template(self, template='conley09f'): 
        """Sets the SED templates that are going to be used for the mangling.
        
        Parameters
        ----------
        template : str, default 'conley09f'
            Template name.
        
        """
        
        path = piscola.__path__[0]
        file = f'{path}/templates/{template}/snflux_1av2.dat'  # v2 is the interpolated version to 0.5 day steps
        self.sed['info'] = pd.read_csv(file, delim_whitespace=True, names=['phase', 'wave', 'flux'])
        self.sed['name'] = template
    
    
    def set_sed_epoch(self, set_eff_wave=True): 
        """Sets the SED phase given the current value of 'self.phase'.
        
        The chosen template is used to do all the corrections. The SED is immediately moved to the SN frame.
        
        Parameters
        ----------
        set_eff_wave : bool, default 'True'
            If True set the effective wavelengths of the filters given the SED at the current phase.
            
        """
        
        sed_data = self.sed['info'][self.sed['info'].phase == self.phase]
        self.sed['wave'], self.sed['flux'] = sed_data.wave.values*(1+self.z), sed_data.flux.values/(1+self.z)
        
        # These fluxes are used in the mangling process to compared to mangled ones with these.
        for band in self.bands:
            flux = run_filter(self.sed['wave'], self.sed['flux'], self.filters[band]['wave'], 
                              self.filters[band]['transmission'], self.filters[band]['response_type'])
            self.sed[band] = {'flux': flux}
        
        # add filter's effective wavelength, which depends on the SED and the phase.
        if set_eff_wave:
            self.set_eff_wave()
        
        
    def set_eff_wave(self):
        """Sets the effective wavelength of each band using the current state of the SED."""
        
        for band in self.filters.keys():
            self.filters[band]['eff_wave'] = calc_eff_wave(self.sed['wave'], self.sed['flux'], 
                                                           self.filters[band]['wave'], self.filters[band]['transmission'], 
                                                           self.filters[band]['response_type'])
        
            
    def plot_sed_state(self, save=False, fig_name=None):
        """Plots the current state of the SED.
        
        Parameters
        ----------
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be '{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """
        
        f, ax = plt.subplots(figsize=(8,6))
        ax.plot(self.sed['wave'], self.sed['flux'])
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=18, family='serif')
        ax.set_ylabel(r'Flux [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', fontsize=18, family='serif')
        ax.set_title(r'SED current state', fontsize=18, family='serif')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.minorticks_on()
        ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)
        
        if save:
            if fig_name is None:
                fig_name = f'{self.name}_sed{self.phase}.png'
            f.tight_layout()
            plt.savefig(f'plots/{fig_name}')
            
        plt.show()
        
        
    # this function might not be necessary for the release of the code   
    def plot_sed_and_filters(self, filter_list=None, save=False, fig_name=None):
        """Plots the current state of the SED.
        
        Parameters
        ----------
        filter_list : list, default 'None'
            List of filters to plot.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be '{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """
        
        if filter_list is None:
            filter_list = self.bands
        
        f, ax = plt.subplots(figsize=(8,6)) 
        ax.plot(self.sed['wave'], self.sed['flux']/self.sed['flux'].max())
            
        limits = np.empty(0)
        for band in filter_list:      
            norm = self.filters[band]['transmission'].max()
            ax.plot(self.filters[band]['wave'], self.filters[band]['transmission']/norm, label=band)
            limits = np.r_[limits, self.filters[band]['wave']]
            
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=18)
        ax.set_ylabel('Relative Flux', fontsize=18)
        ax.set_title(r'SED current state + filters', fontsize=18)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.set_xlim(limits.min(), limits.max())
        ax.set_ylim(0,1)
        ax.minorticks_on()
        ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)
        plt.legend()
        
        if save:
            if fig_name is None:
                fig_name = f'{self.name}_sed{self.phase}_filters.png'
            f.tight_layout()
            plt.savefig(f'plots/{fig_name}')
            
        plt.show()
        
        
    # this function might not be necessary for the release of the code   
    def test_sed_flux(self, verbose=False):
        '''Plots the SED with the integrated fluxes per band.'''
        
        fluxes = {}
        f, ax = plt.subplots(figsize=(8,6))
        ax.plot(self.sed['wave'], self.sed['flux'])
        
        for band in self.filters.keys():
            # check if the filter wavelength is within the permited SED wavelength range
            min_idx, max_idx = filter_effective_range(self.filters[band]['transmission'])
            
            if self.sed['wave'].max() >= self.filters[band]['wave'][max_idx] and self.sed['wave'].min() <= self.filters[band]['wave'][min_idx]: 
                flux_band = run_filter(self.sed['wave'], self.sed['flux'], self.filters[band]['wave'], 
                                       self.filters[band]['transmission'], self.filters[band]['response_type'])
                ax.plot(self.filters[band]['eff_wave'], flux_band, 'o', label=band)
                fluxes.update({band:flux_band})
                
        ax.set_xlabel(r'wavelength ($\AA$)', fontsize=18)
        ax.set_ylabel('flux', fontsize=18)
        ax.set_title(r'SED current state', fontsize=18)
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        plt.legend()
        
        plt.show()
        
        if verbose:
            print('current SED fluxes:\n', fluxes)
        
    #######################################################
    ###################### lcs fit ########################
    #######################################################
            
    def fit_lcs(self, kernel='matern52'):
        """Fits the data for each band using gaussian process
        
        The fits are done independently for each band. The initial B-band peak time is estimated with 
        the pivot band as long as a peak can me calculated, having a derivative equal to zero.
        
        Parameters
        ----------
        kernel : str, default 'matern52'
            Kernel to be used with gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.

        """

        bands_for_pivot = []  # bands with available peaks among which the pivot band will be selected
        # Correct for time dilation for a better fit, but move the data back again in redshift
        for band in self.bands:
            time, flux, std = fit_gp(self.data[band]['mjd'], self.data[band]['flux'], 
                                     self.data[band]['flux_err'], kernel=kernel)
            
            # find index of peak flux
            try: 
                peakidxs = peak.indexes(flux, thres=.3, min_dist=5//(time[1]-time[0])) 
                # pick the index of the first peak in case some band has 2 peaks (like IR bands)
                idx_max = np.asarray([idx for idx in peakidxs if all(flux[:idx]<flux[idx])]).min()  
                tmax = time[idx_max]
                bands_for_pivot.append(band)

            except:
                tmax = np.nan
            self.lc_fits[band] = {'mjd':time, 'flux':flux, 'std':std, 'tmax':tmax}

        assert bands_for_pivot, 'Unable to find a peak! Not enough data around peak luminosity for any band!'
        
        # Check if the available bands cover the B-band wavelength range
        B_covarage = False
        
        self.calc_pivot(bands_for_pivot)
        B_min, B_max = filter_effective_range(self.filters['Bessell_B']['transmission'])

        # bands with coverage around pivot_band peak
        valid_bands = [band for band in self.bands if any(self.lc_fits[band]['mjd'] < self.lc_fits[self.pivot_band]['tmax'])
                                                   and any(self.lc_fits[band]['mjd'] > self.lc_fits[self.pivot_band]['tmax'])]
        
        # indexes for the effective ranges of the bands
        range_indexes = {band:filter_effective_range(self.filters[band]['transmission']) for band in valid_bands}

        blue_edges = np.asarray([(self.filters[band]['wave'][range_indexes[band][0]:]/(1+self.z)).min() for band in valid_bands])
        red_edges = np.asarray([(self.filters[band]['wave'][:range_indexes[band][1]]/(1+self.z)).max() for band in valid_bands])

        # 200 Angstroms is the same tolerance given in GP for extrapolating
        if (any(blue_edges-200 <= self.filters['Bessell_B']['wave'][B_min:].min()) and 
            any(red_edges+200 >= self.filters['Bessell_B']['wave'][:B_max].max()) ):
            B_covarage = True

        assert B_covarage, 'Data need to have better B-band wavelength coverage!'
        
        self.tmax = self.lc_fits[self.pivot_band]['tmax']                               

        
    def plot_fits(self, plot_together=True, save=False, fig_name=None):
        """Plots the light-curves fits results.
        
        Plots the observed data for each band together with the gaussian process fits. The initial B-band
        peak estimation is plotted. The final B-band peak estimation after light-curves corrections is
        also potted if corrections have been applied.
        
        Parameters
        ----------
        plot_together : bool, default 'True'
            If 'True', plots the bands together in one plot. Otherwise, each band is plotted separately.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be ''{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """
        
        if plot_together:
            
            new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]
            
            exp = np.round(np.log10(self.data[self.bands[0]]['flux'].max()), 0)
            y_norm = 10**exp
            
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, band in enumerate(self.bands):

                time, flux, std = np.copy(self.lc_fits[band]['mjd']), np.copy(self.lc_fits[band]['flux']), np.copy(self.lc_fits[band]['std'])
                flux, std = flux/y_norm, std/y_norm
                data_time, data_flux, data_std = np.copy(self.data[band]['mjd']), np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])
                data_flux, data_std = data_flux/y_norm, data_std/y_norm
                
                ax.errorbar(data_time, data_flux, data_std, fmt='o', capsize=3, color=new_palette[i],label=band)
                ax.plot(time, flux,'-', color=new_palette[i])
                ax.fill_between(time, flux-std, flux+std, alpha=0.5, color=new_palette[i])
                
            ax.axvline(x=self.tmax, color='r', linestyle='--')
            ax.axvline(x=self.lc_fits[self.pivot_band]['tmax'], color='k', linestyle='--')
            ax.minorticks_on()
            ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True, labelsize=16)
            ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True, labelsize=16)
            ax.set_xlabel('Modified Julian Date', fontsize=16, family='serif')
            ax.set_ylabel(r'Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp, fontsize=16, family='serif')
            ax.set_title(f'{self.name} (z = {self.z:.5})', fontsize=18, family='serif')
            ax.legend(fontsize=13)
        
        else:
            h = 3
            v = math.ceil(len(self.bands) / h)
        
            fig = plt.figure(figsize=(15, 5*v))
            gs = gridspec.GridSpec(v , h)

            for i, band in enumerate(self.bands):
                j = math.ceil(i % h)
                k =i // h
                ax = plt.subplot(gs[k,j])

                time, flux, std = self.lc_fits[band]['mjd'], self.lc_fits[band]['flux'], self.lc_fits[band]['std']
                ax.errorbar(self.data[band]['mjd'], self.data[band]['flux'], self.data[band]['flux_err'], fmt='ok')
                ax.plot(time, flux,'-')
                ax.fill_between(time, flux-std, flux+std, alpha=0.5)
                
                ax.axvline(x=self.tmax, color='r', linestyle='--')
                ax.axvline(x=self.lc_fits[self.pivot_band]['tmax'], color='k', linestyle='--')
                ax.set_title(f'{band}', fontsize=16, family='serif')
                ax.xaxis.set_tick_params(labelsize=15)
                ax.yaxis.set_tick_params(labelsize=15)
                ax.minorticks_on()
                ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
                ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)

                fig.text(0.5, 0.95, f'{self.name} (z = {self.z:.5})', ha='center', fontsize=20, family='serif')
                fig.text(0.5, 0.04, 'Modified Julian Date', ha='center', fontsize=18, family='serif')
                fig.text(0.04, 0.5, r'Flux [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]', va='center', rotation='vertical', fontsize=18, family='serif')
            
        if save:
            if fig_name is None:
                fig_name = f'{self.name}_lcfits.png'
            fig.tight_layout()
            plt.savefig(f'plots/{fig_name}')
            
        plt.show()
        
    
    #######################################################
    ####################### data ##########################
    #######################################################
    
    def mask_data(self, band_list=None, mask_snr=True, snr=3, mask_phase=False, min_phase=-20, max_phase=40):
        """Mask the data with the given S/N ratio and/or within the given range of days respect to maximum in B band.
        
        NOTE: Bands with less than 3 data points after mask is applied will be deleted.
        
        Parameters
        ----------
        band_list : list, default 'None'
            List of bands to mask. If 'None', the mask is applied to all bands in self.bands
        mask_snr : bool, default 'True'
            If 'True', keeps the flux values with S/N ratio greater or equal to the threshold 'snr'.
        snr : float, default '3'
            S/N ratio threshold applied to mask data.
        mask_phase : bool, default 'False'
            If 'True', keeps the flux values within the given phase range set by 'min_phase' and 'max_phase'.
            The light curves need to be fit first with 'fit_lcs()'.
        min_phase : float, default '-20'
            Minimum phase threshold applied to mask data.
        max_phase : float, default '40'
            Maximum phase threshold applied to mask data.

        """
        
        if band_list is None:
            band_list = self.bands
            
        if mask_snr:        
            for band in band_list:
                mask = np.where(np.abs(self.data[band]['flux']/self.data[band]['flux_err']) >= snr)
                self.data[band]['mjd'] = self.data[band]['mjd'][mask]
                self.data[band]['flux'] = self.data[band]['flux'][mask]
                self.data[band]['flux_err'] = self.data[band]['flux_err'][mask]
                
                if len(self.data[band]['flux']) <= 2:
                    self.delete_bands([band])  # delete bands with less than 3 data points after applying mask
        
        if mask_phase:   
            assert self.lc_fits, 'The light curves need to be fitted first!'
            
            for band in band_list:
                mask = np.where((self.data[band]['mjd'] - self.tmax >= min_phase*(1+self.z)) & 
                                (self.data[band]['mjd'] - self.tmax <= max_phase*(1+self.z))
                               )
                self.data[band]['mjd'] = self.data[band]['mjd'][mask]
                self.data[band]['flux'] = self.data[band]['flux'][mask]
                self.data[band]['flux_err'] = self.data[band]['flux_err'][mask]
        
                if len(self.data[band]['flux']) <= 2:
                    self.delete_bands([band])  # delete bands with less than 3 data points after applying mask
                    
                
    # this function might not be necessary for the release of the code                            
    def integrate_filters(self, band_list=None, value_type='flux'):  
        """Integrate the current state of the SED through the filters.
        
        Parameters
        ----------
        band_list : list, default 'None'
            List of filters to plot. If 'None', band list is set to 'self.filters.keys()'.
        value_type : str, default 'flux'
            Type of data to be returned: either 'flux' or 'mag'.

        """
        
        assert (value_type=='mag' or value_type=='flux'), f'"{value_type}" is not a valid type.'
        
        if band_list is None:
            band_list = self.filters.keys()
            
        idxs = {band:filter_effective_range(self.filters[band]['transmission']) for band in band_list}
        
        if value_type=='flux':
            return {band:run_filter(self.sed['wave'], self.sed['flux'], self.filters[band]['wave'], 
                                    self.filters[band]['transmission'], self.filters[band]['response_type']) 
                    for band in band_list if ( self.sed['wave'].max() >= self.filters[band]['wave'][idxs[band][1]] ) 
                    and ( self.sed['wave'].min() <= self.filters[band]['wave'][idxs[band][0]])}
        
        mags = {band:-2.5*np.log10(run_filter(self.sed['wave'], self.sed['flux'], self.filters[band]['wave'], 
                                    self.filters[band]['transmission'], self.filters[band]['response_type'])) + self.data[band]['zp'] 
                    for band in band_list if band in self.bands and ( self.sed['wave'].max() >= self.filters[band]['wave'][idxs[band][1]] ) 
                    and ( self.sed['wave'].min() <= self.filters[band]['wave'][idxs[band][0]])}
        
        mags.update({band:-2.5*np.log10(run_filter(self.sed['wave'], self.sed['flux'], self.filters[band]['wave'], 
                                    self.filters[band]['transmission'], self.filters[band]['response_type'])) + self.final_results[band]['zp'] 
                    for band in band_list if len(self.final_results[band]['phase'])!=0 and band not in mags.keys() and
                     ( self.sed['wave'].max() >= self.filters[band]['wave'][idxs[band][1]] ) 
                    and ( self.sed['wave'].min() <= self.filters[band]['wave'][idxs[band][0]])})
        
        print(mags)
    
    
    def plot_data(self, band_list=None, plot_type='mag', save=False, fig_name=None):
        """Plot the SN light curves.
        
        Negative fluxes are masked out if magnitudes are plotted.
        
        Parameters
        ----------
        band_list : list, default 'None'
            List of filters to plot. If 'None', band list is set to 'self.bands'.
        plot_type : str, default 'mag'
            Type of value plotted: either 'mag' or 'flux'.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be ''{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """
        
        assert (plot_type=='mag' or plot_type=='flux'), f'"{plot_type}" is not a valid plot type.'
        new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]
        
        if band_list is None:
            band_list = self.bands
        
        exp = np.round(np.log10(self.data[band_list[0]]['flux'].max()), 0)
        y_norm = 10**exp
        
        f, ax = plt.subplots(figsize=(8,6))    
        for i, band in enumerate(band_list):
            if plot_type=='flux':
                time, flux, err = np.copy(self.data[band]['mjd']), np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])
                flux, err = flux/y_norm, err/y_norm
                ax.errorbar(time, flux, err, fmt='o', capsize=3, label=band, color=new_palette[i])
                ylabel = r'Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp
            elif plot_type=='mag':
                ylabel = 'Apparent Magnitude'
                mask = np.where(self.data[band]['flux'] > 0)
                mjd = self.data[band]['mjd'][mask]
                mag = -2.5*np.log10(self.data[band]['flux'][mask]) + self.data[band]['zp']
                err = np.abs(2.5*self.data[band]['flux_err'][mask]/(self.data[band]['flux'][mask]*np.log(10)))
                
                ax.errorbar(mjd, mag, err, fmt='o', capsize=3, label=band, color=new_palette[i])
            
        ax.set_ylabel(ylabel, fontsize=16, family='serif')
        ax.set_xlabel('Modified Julian Date', fontsize=16, family='serif')
        ax.set_title(f'{self.name} (z = {self.z:.5})', fontsize=18, family='serif')
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=16)
        ax.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True, labelsize=16)
        plt.legend(fontsize=13)
        
        if plot_type=='mag':
            plt.gca().invert_yaxis()
            
        if save:
            if fig_name is None:
                fig_name = f'{self.name}_lcs.png'
            f.tight_layout()
            plt.savefig(f'plots/{fig_name}')
            
        plt.show()

        
    def normalize_data(self):
        """Normalize the zero-points.
        
        Zero-points are normalized according to the chosen type of normalization, scaling the fluxes
        accordingly as well. 'ab' uses the AB spectrum to normalize. 'vega' uses 'alpha_lyr_stis_005'.
        'counts' sets all the zero-points to have the same value (arbitrary choice).
            
        """
                    
        for band in self.bands:
            mag_sys = self.data[band]['mag_sys']
            if mag_sys=='VEGA':
                zp = calc_zp_vega(self.filters[band]['wave'], 
                                  self.filters[band]['transmission'], 
                                  self.filters[band]['response_type'])
            elif mag_sys=='AB':
                zp = calc_zp_ab(self.filters[band]['pivot_wave'])
                
            self.data[band]['flux'] = self.data[band]['flux']*10**(-0.4*(self.data[band]['zp'] - zp))
            self.data[band]['flux_err'] = self.data[band]['flux_err']*10**(-0.4*(self.data[band]['zp'] - zp))
            # update the zp only at the end
            self.data[band]['zp'] = zp
        
            
    def delete_bands(self, bands, verbose=False):
        """Delete chosen bands together with the data in it.
        
        Parameters
        ----------
        bands : list
            List of bands.
        verbose : bool, default 'False'
            If 'True', a warning is given when a band from 'bands' was found within the SN bands.
            
        """
        
        for band in bands:
            try:
                self.data.pop(band, None)
                self.bands.remove(band)
            except: 
                if verbose:
                    print('Warning, the chosen bands were not found!')
                
            
    def set_interp_data(self, restframe_phases):
        """Sets the data to be used for the light-curves corrections.
        
        Collects the interpolated data (from the fit) within the phase range given by 'restframe_phases', 
        to later on apply extinction and K-corrections to it.
        
        Parameters
        ----------
        restframe_phases : array
            Array of phases respect to B-band peak.
            
        """

        mjd_range = self.tmax + restframe_phases*(1+self.z)
        
        self.lc_interp = {band:{'flux':np.empty(0), 'flux_err':np.empty(0), 'mjd':np.empty(0), 'phase':np.empty(0)} 
                          for band in self.bands}
        self.lc_correct = {band:{'flux': np.empty(0), 'mag': np.empty(0), 'phase': np.empty(0), 
                                 'flux_err': np.empty(0), 'mag_err': np.empty(0)} 
                              for band in self.filters.keys()}
        self.lc_final_fits = {band:{'flux': np.empty(0), 'mag': np.empty(0), 'phase': np.empty(0), 
                                    'flux_err': np.empty(0), 'mag_err': np.empty(0)} 
                              for band in self.filters.keys()}
        
        for band in self.bands:
            for mjd, phase in zip(mjd_range, restframe_phases):
                if any(np.abs(self.lc_fits[band]['mjd'] - mjd)<0.2):
                    idx = np.argmin(np.abs(self.lc_fits[band]['mjd'] - mjd))
                    self.lc_interp[band]['flux'] = np.r_[self.lc_interp[band]['flux'], self.lc_fits[band]['flux'][idx]]
                    self.lc_interp[band]['flux_err'] = np.r_[self.lc_interp[band]['flux_err'], self.lc_fits[band]['std'][idx]]
                    self.lc_interp[band]['mjd'] = np.r_[self.lc_interp[band]['mjd'], mjd]
                    self.lc_interp[band]['phase'] =  np.r_[self.lc_interp[band]['phase'], phase]
                
            self.lc_interp[band]['tmax'] = self.lc_fits[band]['tmax']
            self.lc_interp[band]['zp'] = self.data[band]['zp']
            self.lc_interp[band]['mag_sys'] = self.data[band]['mag_sys']
            
            
    def plot_interp_data(self, band_list=None, plot_type='flux'):
        """Plot the interpolated data from the SN light curve fit given by the function set_interp_data.
        
        Parameters
        ----------
        band_list : list, default 'None'
            List of bands. If 'None', band list is set to 'self.bands'.
        plot_type: str, default 'flux'
            Type of data to be plotted: either 'mag' or 'flux'.
            
        """
        
        assert (plot_type=='mag' or plot_type=='flux'), f'"{plot_type}" is not a valid plot type.'
        
        new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]
        
        if band_list is None:
            band_list = self.bands
            
        f, ax = plt.subplots(figsize=(8,6))    
        for i, band in enumerate(band_list):
            if plot_type=='flux':
                ax.errorbar(self.lc_interp[band]['phase'], self.lc_interp[band]['flux'], self.lc_interp[band]['flux_err'], 
                            fmt='o', label=band, color=new_palette[i])
                ylabel = r'Flux [erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'
            elif plot_type=='mag':
                ylabel = 'Apparent Magnitude'
                mjd = self.lc_interp[band]['phase']
                mag = -2.5*np.log10(self.lc_interp[band]['flux']) + self.lc_interp[band]['zp']
                err = np.abs(2.5*self.lc_interp[band]['flux_err']/(self.lc_interp[band]['flux']*np.log(10)))
                
                ax.errorbar(mjd, mag, err, fmt='o', label=band, color=new_palette[i])
        
        ax.set_ylabel(ylabel, fontsize=16, family='serif')
        ax.set_xlabel('Phase', fontsize=16, family='serif')
        ax.axvline(x=0, color='k', linestyle='--')
        ax.set_title(f'{self.name} (z = {self.z:.5})', fontsize=18, family='serif')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True)
        ax.legend(fontsize=13)
        
        if plot_type=='mag':
            plt.gca().invert_yaxis()
            
        plt.show()
        
    
    #######################################################
    ############## light curve corrections ################
    #######################################################
    
    def mangle_sed(self, kernel='squaredexp', method='gp'):
        """Mangles the SED with the given method to match the SN magnitudes.
        
        Parameters
        ----------
        kernel : str, default 'squaredexp'
            Kernel to be used for the gaussian process fit.  Possible choices are: 'matern52', 
            'matern32', 'squaredexp'.
        method: str, default 'gp'
            Method to mangle the SED: either 'gp' for gaussian process or 'spline' for spline.
            NOTE: 'spline' method does not return correct errors, this needs to be fixed in the future.
            
        """
        
        valid_bands = [band for band in self.bands if self.phase in self.lc_interp[band]['phase']] # bands with data at the given phase
        
        # I really need to think of a better way of doing this (nasty while loops)
        #mangling_not_ready = True
        #while mangling_not_ready:
        idx_bands = [np.where(self.lc_interp[band]['phase'] == self.phase)[0][0] for band in valid_bands]

        flux_ratios = np.asarray([self.lc_interp[band]['flux'][idx]/self.sed[band]['flux'] for band, idx 
                                                                                      in zip(valid_bands, idx_bands)])
        flux_ratios_err = np.asarray([self.lc_interp[band]['flux_err'][idx]/self.sed[band]['flux'] for band, idx 
                                                                                      in zip(valid_bands, idx_bands)])

        obs_fluxes = np.asarray([self.lc_interp[band]['flux'][idx] for band, idx in zip(valid_bands, idx_bands)])
        obs_fluxes_err = np.asarray([self.lc_interp[band]['flux_err'][idx] for band, idx in zip(valid_bands, idx_bands)])

        results = mangle(flux_ratios, flux_ratios_err, self.sed['wave'], self.sed['flux'], 
                         valid_bands, self.filters, obs_fluxes, obs_fluxes_err, kernel=kernel, method=method)

        mangled_wave, mangled_flux, mangled_flux_err, mangling_results = results

        mag_diffs = {}
        diff_array = np.empty(0)
        for band, obs_flux in zip(valid_bands, obs_fluxes):
            band_wave, band_transmission = self.filters[band]['wave'], self.filters[band]['transmission']
            response_type = self.filters[band]['response_type']
            model_flux = run_filter(mangled_wave, mangled_flux, band_wave, band_transmission, response_type)

            mag_diffs[band] = -2.5*np.log10(obs_flux) - (-2.5*np.log10(model_flux))
            diff_array = np.r_[diff_array, mag_diffs[band]]
        
        self.sed['wave'], self.sed['flux'], self.sed['flux_err'] = mangled_wave, mangled_flux, mangled_flux_err
        self.mangling_results.update({self.phase:mangling_results})
        self.mangling_results[self.phase].update({'mag_diff':mag_diffs})
        
        self.set_eff_wave()
            
            
    def plot_mangling_function(self, phase=None, mangle_only=True, verbose=True, save=False, fig_name=None):
        """Plot the mangling function for a given phase.
        
        Parameters
        ----------
        band_list : list, default 'None'
            List of filters to plot. If 'None', band list is set to 'self.bands'.
        mangle_only : bool, default 'True'
            If 'True', only plots the mangling function, else, plots the SEDs and filters as well (in a 
            relative scale).
        verbose : bool, default 'True'
            If 'True', returns the difference between the magnitudes from the fits and the magnitudes from the
            modified SED after mangling, for each of the bands in 'band_list'.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be ''{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """
        
        if phase is None:
            phase = self.phase
            
        assert (phase in self.mangling_results.keys()), f'phase {phase} does not have a mangling result.'
            
        man = self.mangling_results[phase]
        eff_waves = np.copy(man['init_vals']['waves'])
        init_flux_ratios = np.copy(man['init_vals']['flux_ratios'])
        flux_ratios_err = np.copy(man['init_vals']['flux_ratios_err'])
        
        opt_flux_ratios = np.copy(man['opt_vals']['flux_ratios'])
        obs_fluxes = np.copy(man['obs_vals']['fluxes'])
        sed_fluxes = np.copy(man['sed_vals']['fluxes'])
        
        x, y, yerr = np.copy(man['opt_fit']['waves']), np.copy(man['opt_fit']['flux_ratios']), np.copy(man['opt_fit']['flux_ratios_err'])
        mang_sed_wave, mang_sed_flux = man['mangled_sed']['wave'], man['mangled_sed']['flux']
        init_sed_wave, init_sed_flux = man['init_sed']['wave'], man['init_sed']['flux']
        
        kernel = man['kernel']
        bands = man['mag_diff'].keys()
        
        if mangle_only:
            f, ax = plt.subplots(figsize=(8,6))
            ax2 = ax.twiny()
            
            exp = np.round(np.log10(init_flux_ratios.max()), 0)
            y_norm = 10**exp
            init_flux_ratios, flux_ratios_err = init_flux_ratios/y_norm, flux_ratios_err/y_norm
            y, yerr = y/y_norm, yerr/y_norm
            opt_flux_ratios = opt_flux_ratios/y_norm
            
            ax.errorbar(eff_waves, init_flux_ratios, flux_ratios_err, fmt='o', capsize=3, label='Initial values')
            ax.plot(x, y)
            ax.fill_between(x, y-yerr, y+yerr, alpha=0.5, color='orange')
            ax.errorbar(eff_waves, opt_flux_ratios, flux_ratios_err, fmt='*', capsize=3, color='red', label='Optimized values')

            ax.set_xlabel(r'Observer-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax.set_ylabel(r'Flux$_{\rm Obs}$ / Flux$_{\rm Temp} \times$ 10$^{%.0f}$'%exp, fontsize=16, family='serif')
            ax.set_title(f'Mangling Function', fontsize=18, family='serif')
            ax.minorticks_on()
            ax.tick_params(which='major', length=8, width=1, direction='in', right=True, labelsize=16)
            ax.tick_params(which='minor', length=4, width=1, direction='in', right=True, labelsize=16)

            xticks = x[::len(x)//(len(self.bands)+1)]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(np.round(xticks/(1+self.z), 0))
            ax2.set_xlabel(r'Rest-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax2.tick_params(which='major', length=8, width=1, direction='in', labelsize=16)
            ax2.tick_params(which='minor', length=4, width=1, direction='in', labelsize=16)

            for i, band in enumerate(bands):
                x1, y1 = ax.transLimits.transform((eff_waves[i], init_flux_ratios[i]))
                ax.text(x1, y1+(-1)**i*0.12, band, horizontalalignment='center', 
                        verticalalignment='center', transform=ax.transAxes, fontsize=14)
            ax.legend(loc='upper right', fontsize=12)
            
        else:
            f, ax = plt.subplots(figsize=(8,6))
            ax2 = ax.twiny()
            ax3 = ax.twinx()

            norm = 2  # for bands
            norm2 = 1  # for SEDs
            index = (len(bands)-1)//2  # index of the band to do relative comparison
            
            init_sed_flux2 = init_sed_flux/sed_fluxes[index]
            mang_sed_flux2 = mang_sed_flux/obs_fluxes[index]
            sed_fluxes2 =sed_fluxes/sed_fluxes[index]
            obs_fluxes2 = obs_fluxes/obs_fluxes[index]
            
            # filters
            for i, band in enumerate(bands):
                wave, trans = self.filters[band]['wave'], self.filters[band]['transmission']
                if i==index:
                    ax3.plot(wave, trans/trans.max()*norm, color='b', alpha=0.4)
                else:
                    ax3.plot(wave, trans/trans.max()*norm, color='k', alpha=0.4)
                
            # mangling function        
            ax.plot(x, y/opt_flux_ratios[index], 'green')  
            indexes = [np.argmin(np.abs(x-wave_val)) for wave_val in eff_waves]
            #ax.plot(eff_waves, opt_flux_ratios/opt_flux_ratios[index], 'sg')
            ax.plot(eff_waves, y[indexes]/opt_flux_ratios[index], 'sg')
                
            # initial sed and fluxes
            ax3.plot(init_sed_wave, init_sed_flux2*norm2, '--k')  # initial sed
            ax3.plot(eff_waves, sed_fluxes2*norm2, 'ok', label='Initial values', alpha=0.8, fillstyle='none')  # initial sed fluxes
            
            # optimized sed and fluxes
            ax3.plot(mang_sed_wave, mang_sed_flux2*norm2, 'red')  # mangled sed
            ax3.plot(eff_waves, obs_fluxes2*norm2,'*r', label='Observed values')  # observed fluxes

            ax.set_xlabel(r'Observer-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax.set_ylabel(r'Relative Mangling Function', fontsize=16, family='serif', color='g')
            ax.set_title(f'Mangling Function', fontsize=18, family='serif')
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim((y/opt_flux_ratios[index]).min()*0.8, (y/opt_flux_ratios[index]).max()*1.2)
            ax.minorticks_on()
            ax.tick_params(which='major', length=8, width=1, direction='in', labelsize=16)
            ax.tick_params(which='minor', length=4, width=1, direction='in', labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16, labelcolor='g')

            xticks = x[::len(x)//(len(self.bands)+1)]
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(xticks)
            ax2.set_xticklabels(np.round(xticks/(1+self.z), 0))
            ax2.set_xlabel(r'Rest-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax2.tick_params(which='major', length=8, width=1, direction='in', labelsize=16)
            ax2.tick_params(which='minor', length=4, width=1, direction='in', labelsize=16)
            ax2.xaxis.set_tick_params(labelsize=16)
            
            ax3.set_ylim(0, None)
            ax3.yaxis.set_tick_params(labelsize=16)
            ax3.set_ylabel(r'Relative Flux', fontsize=16, family='serif', rotation=270, labelpad=20)
            ax3.tick_params(which='major', length=8, width=1, direction='in', labelsize=16)
            ax3.tick_params(which='minor', length=4, width=1, direction='in', labelsize=16)

            ax3.legend(loc='upper right', fontsize=12)
        
        if save:
            if fig_name is None:
                fig_name = f'{self.name}_mangling_phase{phase}.png'
            f.tight_layout()
            plt.savefig(f'plots/{fig_name}')
            
        plt.show()
        
        if verbose:
            print(f'Mangling results, i.e., difference between mangled SED and "observed" magnitudes, at phase {phase}:')
            for band, diff in man['mag_diff'].items():      
                print(f'{band}: {diff} [mags]')
    
    
    def correct_extinction(self, scaling=0.86):
        """Corrects the SED for extinction.
        
        Corrects the SED for extinction using a given extinction law. The default is the
        Cardelli, Clayton & Mathis (1989) extinction law; however, this can be changed.
        Check the extinction package Docs for other options. The scaling refers to the 
        recalibration of the Schlegel, Finkbeiner & Davis (1998) dust map. Check the 
        'scaling' parameter below.
        
        Parameters
        ----------
        scaling : float, default '0.86'
            Recalibration of the Schlegel, Finkbeiner & Davis (1998) dust map. Either '0.86'
            for the Schlafly & Finkbeiner (2011) recalibration or '1.0' for the original 
            dust map of Schlegel, Finkbeiner & Davis (1998).
        
        """
            
        self.sed['flux'] = deredden(self.sed['wave'], self.sed['flux'], self.ra, self.dec, scaling=scaling)
        self.sed['flux_err'] = deredden(self.sed['wave'], self.sed['flux_err'], self.ra, self.dec, scaling=scaling)
        
        self.set_eff_wave()

    
    def kcorrection(self):
        """Applies K-corrections.
        
        The K-corrections is not calculated explicitly. This function moves the SED in redshift and then
        calculates the fluxes and magnitudes at rest-frame.
        
        """
        
        self.sed['wave'], self.sed['flux'], self.sed['flux_err'] = self.sed['wave']/(1+self.z), self.sed['flux']*(1+self.z), self.sed['flux_err']*(1+self.z)
                        
        for band in self.filters.keys():
            # check if the filter wavelength is within the permited SED wavelength range
            min_idx, max_idx = filter_effective_range(self.filters[band]['transmission'])
            if max_idx == len(self.filters[band]['wave']):
                max_idx = -1  # to prevent indexing issues
            if (self.sed['wave'].max() >= self.filters[band]['wave'][max_idx]) and (self.sed['wave'].min() <= self.filters[band]['wave'][min_idx]):
                
                if band in self.bands:
                    zp = self.data[band]['zp']
                    mag_sys = self.data[band]['mag_sys']
                else: 
                    # only for Bessell filters
                    zp = calc_zp_vega(self.filters[band]['wave'], self.filters[band]['transmission'], self.filters[band]['response_type'])
                    mag_sys = 'VEGA'    
                                   
                band_flux = run_filter(self.sed['wave'], self.sed['flux'], 
                                       self.filters[band]['wave'], self.filters[band]['transmission'], self.filters[band]['response_type'])
                band_flux_err = run_filter(self.sed['wave'], self.sed['flux_err'], 
                                           self.filters[band]['wave'], self.filters[band]['transmission'], self.filters[band]['response_type'])
                band_mag = -2.5*np.log10(band_flux) + zp
                band_mag_err = np.abs(2.5*band_flux_err/(band_flux*np.log(10)))

                self.lc_correct[band]['flux'] = np.r_[self.lc_correct[band]['flux'], band_flux]
                self.lc_correct[band]['flux_err'] = np.r_[self.lc_correct[band]['flux_err'], band_flux_err]
                self.lc_correct[band]['mag'] = np.r_[self.lc_correct[band]['mag'], band_mag]
                self.lc_correct[band]['mag_err'] = np.r_[self.lc_correct[band]['mag_err'], band_mag_err]
                self.lc_correct[band]['phase'] = np.r_[self.lc_correct[band]['phase'], self.phase]
                self.lc_correct[band]['zp'] = zp
                self.lc_correct[band]['mag_sys'] = mag_sys
                
        self.set_eff_wave()

                
    def correct_light_curve(self, scaling=0.86, **mangle_kwargs):
        """Applies correction to the light curves.
        
        Runs the 'mangle_sed()', 'correct_extinction()' and 'kcorrection()' functions on the SN data,
        for every phase given in 'set_interp_data()'.
        
        Parameters
        ----------
        scaling : float, default '0.86'
            Check 'correct_extinction()' for more information.
        **mangle_kwargs : 
            Check 'mangle_sed()' for more information.

        """
        
        if 'kernel' in mangle_kwargs.keys():
            assert (mangle_kwargs['kernel']=='matern52' or mangle_kwargs['kernel']=='matern32'
                    or mangle_kwargs['kernel']=='squaredexp'or mangle_kwargs['kernel']==None), f'"{mangle_kwargs["kernel"]}" is not a valid kernel.'
            
        #print(f'Starting light curve correction for {self.name}...')
        for phase in self.lc_interp[self.pivot_band]['phase']:
                
            self.phase = phase
            self.set_sed_epoch()
            
            try:
                self.mangle_sed(**mangle_kwargs)        # first mangle to correct the effective wavelength estimation...
                self.set_sed_epoch(set_eff_wave=False)  # reset SED with better estimation (from 1st mangling) of effective wavelengths...
                self.mangle_sed(**mangle_kwargs)        # mangle again
            except: 
                #print(f'Warning, mangling in phase {phase} failed for {self.name}!')
                if phase in self.mangling_results:
                    del self.mangling_results[phase]
            else:
                self.correct_extinction(scaling=scaling)
                self.kcorrection()     
                self.sed_results.update({self.phase:{'wave':self.sed['wave'], 'flux':self.sed['flux'], 
                                                               'flux_err':self.sed['flux_err']}})
                
                        
    def check_B_peak(self, threshold=0.2, iter_num=1, maxiter=5, scaling=0.86, **mangle_kwargs):
        """Estimate the B-band peak from the corrected light curves.
        
        Finds B-band peak and compares it with the initial value. If they do not match within the given threshold, 
        re-do the light-curves correction process with the new estimated peak. This whole process is done a several
        times until the threshold or the maximum number of iteration is reached, whichever comes first.
        
        Parameters
        ----------
        threshold : float, default '0.2'
            Threshold for the difference between the initial B-band peak estimation and the new estimation.
        iter_num : int, default '1'
            This value counts the number of iteration for the light-curves correction process.
        maxiter : int, default '5'
            Maximum number of iterations.
        scaling : float, default '0.86'
            Check 'correct_extinction()' for more information.
        **mangle_kwargs : 
            Check 'mangle_sed()' for more information.

        """
        
        B_band = 'Bessell_B'
        phase, flux, _ = fit_gp(self.lc_correct[B_band]['phase'], self.lc_correct[B_band]['flux'], self.lc_correct[B_band]['flux_err'])
        
        try:
            # use interpolated data
            peakidxs = peak.indexes(flux, thres=.3, min_dist=5//(phase[1]-phase[0]))
            # pick the index of the first peak in case some band has 2 peaks (like IR bands)
            idx_max = np.asarray([idx for idx in peakidxs if all(flux[:idx]<flux[idx])]).min()
            phase_max = phase[idx_max]  # this needs to be as close to zero as possible
            
            # use discrete data
            #peakidxs = peak.indexes(self.lc_correct[B_band]['flux'], thres=.3, min_dist=5//(self.lc_correct[B_band]['phase'][1]-self.lc_correct[B_band]['phase'][0]))
            #idx_max = np.asarray([idx for idx in peakidxs if all(self.lc_correct[B_band]['flux'][:idx]<self.lc_correct[B_band]['flux'][idx])]).min()
            #phase_max = self.lc_correct[B_band]['phase'][idx_max]  # this needs to be as close to zero as possible
        except:
            raise ValueError(f'Unable to obtain an accurate B-band peak for {self.name}!')
            
        if (iter_num <= maxiter) and (np.abs(phase_max) > threshold):
            self.tmax += phase_max*(1+self.z) 
            #print(f'{self.name} iteration number {iter_num}')
            self.set_interp_data(restframe_phases=self.lc_correct[B_band]['phase'])  # set interpolated data with new tmax
            self.correct_light_curve(scaling=scaling, **mangle_kwargs)
            self.check_B_peak(threshold=threshold, iter_num=iter_num+1, maxiter=maxiter, scaling=scaling, **mangle_kwargs)
            
        elif iter_num == maxiter:
            raise ValueError(f'Unable to constrain B-band peak for {self.name}!')
            
        else:
            self.lc_parameters['phase_max'] = phase_max
                
                
    def calc_lc_parameters(self):
        """Calculates the SN light curve parameters. 
        
        Estimation of B-band peak apparent magnitude (mb), stretch (dm15) and color (Bmax-Vmax) parameters.
        An interpolation of the corrected light curves is done as well as part of this process.
        
        """
        
        B_band = 'Bessell_B' 
        zpB = self.lc_correct[B_band]['zp']
        
        # mb
        try:
            #phaseB, fluxB, flux_errB = fit_gp(self.lc_correct[B_band]['phase'], self.lc_correct[B_band]['flux'], self.lc_correct[B_band]['flux_err'])
            phaseB, fluxB, flux_errB = self.lc_correct[B_band]['phase'], self.lc_correct[B_band]['flux'], self.lc_correct[B_band]['flux_err']
            
            idx_Bmax = np.argmin(np.abs(phaseB))
            mb = -2.5*np.log10(fluxB[idx_Bmax]) + zpB
            dmb = np.abs(2.5*flux_errB[idx_Bmax]/(fluxB[idx_Bmax]*np.log(10)))
        except:
            mb = dmb = np.nan
            
        # stretch    
        try:
            if any(np.abs(phaseB-15.) <= 0.2):
                idx_B15 = np.argmin(np.abs(phaseB-15.))
                B15 = -2.5*np.log10(fluxB[idx_B15]) + zpB
                B15_err = np.abs(2.5*flux_errB[idx_B15]/(fluxB[idx_B15]*np.log(10)))
                dm15 = B15 - mb
                dm15err = np.sqrt(dmb**2 + B15_err**2) 
            else:
                dm15 = dm15err = np.nan
        except:
            dm15 = dm15err = np.nan
            
        # colour
        try:
            V_band = 'Bessell_V'
            zpV = self.lc_correct[V_band]['zp']
            #phaseV, fluxV, flux_errV = fit_gp(self.lc_correct[V_band]['phase'], self.lc_correct[V_band]['flux'], self.lc_correct[V_band]['flux_err'])
            phaseV, fluxV, flux_errV = self.lc_correct[V_band]['phase'], self.lc_correct[V_band]['flux'], self.lc_correct[V_band]['flux_err']
            
            if any(np.abs(phaseV) <= 0.2):
                idx_V0 = np.argmin(np.abs(phaseV))
                V0 = -2.5*np.log10(fluxV[idx_V0]) + zpV
                V0err = np.abs(2.5*flux_errV[idx_V0]/(fluxV[idx_V0]*np.log(10)))
                color = mb - V0
                dcolor = np.sqrt(dmb**2 + V0err**2)  
            else:
                color = dcolor = np.nan
        except:
            color = dcolor = np.nan
        
        self.lc_parameters = {'mb':mb, 'dmb':dmb, 'dm15':dm15, 
                              'dm15err':dm15err, 'color':color, 'dcolor':dcolor}
        
        for band in self.filters.keys():   
            try: # is this "try" even necessary?
                fit_gp_results = fit_gp(self.lc_correct[band]['phase'], 
                                        self.lc_correct[band]['flux'], 
                                        self.lc_correct[band]['flux_err'])
                self.lc_final_fits[band]['phase'] = fit_gp_results[0]
                self.lc_final_fits[band]['flux'] = fit_gp_results[1]
                self.lc_final_fits[band]['flux_err'] = fit_gp_results[2]
                self.lc_final_fits[band]['zp'] = self.lc_correct[band]['zp']
                self.lc_final_fits[band]['mag_sys'] = self.lc_correct[band]['mag_sys']
            except:
                pass
       
    
    def display_results(self, band=None, save=False, fig_name=None):
        """Displays the rest-frame light curve for the given band. 
        
        Plots the rest-frame band light curve together with a gaussian fit to it. The parameters estimated with
        'calc_lc_parameters()' are shown as well.
        
        Parameters
        ----------
        band : str, default 'None'
            Name of the band to be plotted. If 'None', band is set to 'Bessell_B'.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be ''{self.name}_sed{self.phase}.png'.
            Only works if 'save' is set to 'True'.

        """
        
        mb = self.lc_parameters['mb']
        dmb = self.lc_parameters['dmb']
        color = self.lc_parameters['color']
        dcolor = self.lc_parameters['dcolor']
        dm15 = self.lc_parameters['dm15']
        dm15err = self.lc_parameters['dm15err']

        if band is None:
            band = 'Bessell_B'
            
        x = np.copy(self.lc_correct[band]['phase'])
        y = np.copy(self.lc_correct[band]['flux'])
        yerr = np.copy(self.lc_correct[band]['flux_err'])
        x_fit = np.copy(self.lc_final_fits[band]['phase'])
        y_fit = np.copy(self.lc_final_fits[band]['flux'])
        yerr_fit = np.copy(self.lc_final_fits[band]['flux_err'])
        
        exp = np.round(np.log10(y.max()), 0)
        y_norm = 10**exp
        y /= y_norm
        yerr /= y_norm
        y_fit /= y_norm
        yerr_fit /= y_norm

        f, ax = plt.subplots(figsize=(8,6))
        ax.errorbar(x, y, yerr, fmt='-.o', capsize=3, color='k')
        ax.plot(x_fit, y_fit, 'r-', alpha=0.5)
        ax.fill_between(x_fit, y_fit+yerr_fit, y_fit-yerr_fit, alpha=0.5, color='r')

        ax.text(0.75, 0.9,r'm$_B^{\rm max}$=%.3f$\pm$%.3f'%(mb, dmb), ha='center', va='center', fontsize=15, transform=ax.transAxes)
        ax.text(0.75, 0.8,r'$\Delta$m$_{15}$($B$)=%.3f$\pm$%.3f'%(dm15, dm15err), ha='center', va='center', fontsize=15, transform=ax.transAxes)
        ax.text(0.75, 0.7,r'($B-V$)$_{\rm max}$=%.3f$\pm$%.3f'%(color, dcolor), ha='center', va='center', fontsize=15, transform=ax.transAxes)

        ax.set_xlabel(f'Phase with respect to B-band peak [days]', fontsize=16, family='serif')
        ax.set_ylabel('Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp, fontsize=16, family='serif')
        ax.set_title(f'{self.name} ({band}, z={self.z:.5})', fontsize=16, family='serif')
        ax.set_ylim(y.min()*0.90, y.max()*1.05)
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=16)
        ax.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True, labelsize=16)
        
        if save:
            if fig_name is None:
                fig_name = f'{self.name}_{band}_results.png'
            f.tight_layout()
            plt.savefig(f'plots/{fig_name}')
            
        plt.show()
