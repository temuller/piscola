
import piscola
from .filter_utils import integrate_filter, calc_eff_wave, calc_pivot_wave, calc_zp, filter_effective_range
from .gaussian_process import gp_lc_fit, gp_2d_fit
from .extinction_correction import redden, deredden
from .mangling import mangle
from .utils import trim_filters, flux2mag, mag2flux

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

def initialise_sn(sn_file):
    """Initialise the 'sn' object.

    The object is initialised with all the necessary information like filters, fluxes, etc.

    Parameters
    ----------
    sn_file : str
        Name of the SN or SN file.

    Returns
    -------
    New 'sn' object.

    """

    name, z, ra, dec = pd.read_csv(sn_file, delim_whitespace=True, nrows=1).iloc[0].values
    sn_df = pd.read_csv(sn_file, delim_whitespace=True, skiprows=2)
    sn_df.columns = sn_df.columns.str.lower()

    # call sn object
    sn_obj = sn(name, z=z, ra=ra, dec=dec)
    sn_obj.set_sed_template()  # Set the SED template to be used in the entire process
    sn_obj.bands = [band for band in list(sn_df['band'].unique())]
    sn_obj.call_filters()

    # order bands by effective wavelength (estimated from the SED template)
    eff_waves = [sn_obj.filters[band]['eff_wave'] for band in sn_obj.bands]
    sorted_idx = sorted(range(len(eff_waves)), key=lambda k: eff_waves[k])
    sn_obj.bands = [sn_obj.bands[x] for x in sorted_idx]

    # add data to each band
    for band in sn_obj.bands:
        band_info = sn_df[sn_df['band']==band]
        if len(band_info['flux'].values) >= 3:
            sn_obj.data[band] = {'mjd':band_info['mjd'].values,
                                 'flux':band_info['flux'].values,
                                 'flux_err':band_info['flux_err'].values,
                                 'zp':float(band_info['zp'].unique()[0]),
                                 'mag_sys':band_info['mag_sys'].unique()[0],
                                }
    sn_obj.bands = list(sn_obj.data.keys())  # to exclude removed bands
    #sn_obj.calc_pivot()
    return sn_obj


def call_sn(sn_file, directory='data'):
    """Loads a supernova from a file.

    Parameters
    ----------
    sn_file: str
        Name of the SN or SN file.
    directory : str, default 'data/'
        Directory where to look for the SN file unless the full or relative path is given in 'sn_file'.

    """

    # if sn_file is the file name
    if os.path.isfile(os.path.join(directory, sn_file)):
        return initialise_sn(os.path.join(directory, sn_file))

    # if sn_file is the SN name
    elif os.path.isfile(os.path.join(directory, sn_file) + '.dat'):
        return initialise_sn(os.path.join(directory, sn_file) + '.dat')

    # if sn_file is the file name with full or relative path
    elif os.path.isfile(sn_file):
        return initialise_sn(sn_file)

    else:
        raise ValueError(f'{str} was not a valid SN name or file.')

def load_sn(name, path=None):
    """Loads a 'sn' oject that was previously saved as a pickle file.

    Parameters
    ----------
    name : str
        Name of the SN object.
    path: str
        Path where to save the SN file given the 'name'.

    Returns
    -------
    'sn' object previously saved as a pickle file.

    """

    if path is None:
        with open(f'{name}.pisco', 'rb') as file:
            return pickle.load(file)
    else:
        with open(os.path.join(path, name) + '.pisco', 'rb') as file:
            return pickle.load(file)

################################################################################
################################################################################
################################################################################

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
        self.__dict__['lc_parameters'] = {}  # final SN light-curves parameters
        self.__dict__['sed_results'] = {}  # final SED for every phase if successful
        self.__dict__['mangling_results'] = {}  # mangling results for every phase if successful
        self.__dict__['user_input'] = {}  # save user's input
        self.bands = None
        self.tmax = None


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

    def save_sn(self, name=None, path=None):
        """Saves a SN object into a pickle file

        Parameters
        ----------
        name : str
            Name of the SN object. If no name is given, the name is set to 'self.name'.
        path: str
            Path where to save the SN file given the 'name'.
        """

        if name is None:
            name = self.name

        if path is None:
            with open(f'{name}.pisco', 'wb') as pfile:
                pickle.dump(self, pfile, pickle.HIGHEST_PROTOCOL)
        else:
            with open(os.path.join(path, name) + '.pisco', 'wb') as pfile:
                pickle.dump(self, pfile, pickle.HIGHEST_PROTOCOL)


    ############################################################################
    ################################ Filters ###################################
    ############################################################################

    def call_filters(self):
        """Obtains the filters's transmission function for the observed bands and the Bessell bands."""

        path = piscola.__path__[0]
        sed_df = self.sed['info']
        sed_df = sed_df[sed_df.phase==0.0]
        sed_wave, sed_flux = sed_df.wave.values, sed_df.flux.values

        # add filters of the observed bands
        for band in self.bands:
            file = f'{band}.dat'

            for root, dirs, files in os.walk(os.path.join(path, 'filters')):
                if file in files:
                    wave0, transmission0 = np.loadtxt(os.path.join(root, file)).T
                    # linearly interpolate filters
                    wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
                    transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                    # remove long tails of zero values on both edges
                    imin, imax = trim_filters(transmission)
                    wave, transmission = wave[imin:imax], transmission[imin:imax]

                    response_type = 'photon'
                    self.filters[band] = {'wave':wave,
                                          'transmission':transmission,
                                          'eff_wave':calc_eff_wave(sed_wave, sed_flux, wave,
                                                                   transmission, response_type=response_type),
                                          'response_type':response_type}

        # add Bessell filters
        file_paths = [file for file in glob.glob(os.path.join(path, 'filters/Bessell/*.dat'))]

        for file_path in file_paths:
            band = os.path.basename(file_path).split('.')[0]
            wave0, transmission0 = np.loadtxt(file_path).T
            # linearly interpolate filters
            wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
            transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
            # remove long tails of zero values on both edges
            imin, imax = trim_filters(transmission)
            wave, transmission = wave[imin:imax], transmission[imin:imax]

            response_type = 'photon'
            self.filters[band] = {'wave':wave,
                                  'transmission':transmission,
                                  'eff_wave':calc_eff_wave(sed_wave, sed_flux, wave, transmission,
                                                           response_type=response_type),
                                  'response_type':response_type}


    def add_filters(self, filter_list, response_type='photon'):
        """Add choosen filters. You can add a complete directory with filters in it or add filters given in a list.

        Parameters
        ----------
        filter_list : list
            List of bands.
        response_type : str, default 'photon'
            Response type of the filter. The only options are: 'photon' and 'energy'.

        """

        path = piscola.__path__[0]
        sed_df = self.sed['info']
        sed_df = sed_df[sed_df.phase==0.0]
        sed_wave, sed_flux = sed_df.wave.values, sed_df.flux.values

        filters_path = os.path.join(path, 'filters')
        filters_sub_path = os.path.join(filters_path, filter_list)
        if isinstance(filter_list, str) and os.path.isdir(filters_sub_path):
            # add directory
            for file in os.listdir(filters_sub_path):
                if file[-4:]=='.dat':
                    band = file.split('.')[0]
                    wave0, transmission0 = np.loadtxt(os.path.join(filters_sub_path, file)).T
                    # linearly interpolate filters
                    wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
                    transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                    # remove long tails of zero values on both edges
                    imin, imax = trim_filters(transmission)
                    wave, transmission = wave[imin:imax], transmission[imin:imax]

                    self.filters[band] = {'wave':wave,
                                          'transmission':transmission,
                                          'eff_wave':calc_eff_wave(sed_wave, sed_flux, wave, transmission,
                                                                   response_type=response_type),
                                          'response_type':response_type}

        else:
            # add filters in list
            for band in filter_list:
                file = f'{band}.dat'

                for root, dirs, files in os.walk(os.path.join(path, 'filters')):
                    if file in files:
                        wave0, transmission0 = np.loadtxt(os.path.join(root, file)).T
                        # linearly interpolate filters
                        wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max()-wave0.min()))
                        transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                        # remove long tails of zero values on both edges
                        imin, imax = trim_filters(transmission)
                        wave, transmission = wave[imin:imax], transmission[imin:imax]

                        self.filters[band] = {'wave':wave,
                                              'transmission':transmission,
                                              'eff_wave':calc_eff_wave(sed_wave, sed_flux, wave, transmission,
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

        fig, ax = plt.subplots(figsize=(8,6))
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
            fig.tight_layout()
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

        BessellB_eff_wave = self.filters['Bessell_B']['eff_wave']

        if band_list is None:
            band_list = self.bands

        bands_eff_wave =  np.array([self.filters[band]['eff_wave']/(1+self.z) for band in band_list])
        idx = (np.abs(BessellB_eff_wave - bands_eff_wave)).argmin()
        self.pivot_band = band_list[idx]


    def delete_bands(self, bands_list, verbose=False):
        """Delete chosen bands together with the data in it.

        Parameters
        ----------
        bands : list
            List of bands.
        verbose : bool, default 'False'
            If 'True', a warning is given when a band from 'bands' is not found within the SN bands.

        """

        for band in bands_list:
            if band in self.bands:
                self.data.pop(band, None)
                self.filters.pop(band, None)
                self.bands.remove(band)
            else:
                if verbose:
                    print(f'Warning, {band} not found!')

    ############################################################################
    ############################### SED template ###############################
    ############################################################################

    def print_sed_templates(self):
        """Prints all the available SED templates in the 'templates' directory"""

        path = piscola.__path__[0]
        template_path = os.path.join(path, "templates")
        print('The list of available SED templates are:', [name for name in os.listdir(template_path)
                                                           if os.path.isdir(os.path.join(template_path, f"{name}"))])


    def set_sed_template(self, template='jla'):
        """Sets the SED templates that are going to be used for the mangling.

        Parameters
        ----------
        template : str, default 'jla'
            Template name. E.g., 'jla', 'conley09f', etc..

        """
        # This can be modified to accept other templates
        path = piscola.__path__[0]
        file = os.path.join(path, f'templates/{template}/snflux_1a.dat')
        self.sed['info'] = pd.read_csv(file, delim_whitespace=True, names=['phase', 'wave', 'flux'])
        self.sed['name'] = template


    def set_eff_wave(self):
        """Sets the effective wavelength of each band using the current state of the SED."""

        for band in self.filters.keys():
            self.filters[band]['eff_wave'] = calc_eff_wave(self.sed['wave'], self.sed['flux'],
                                                           self.filters[band]['wave'], self.filters[band]['transmission'],
                                                           self.filters[band]['response_type'])

    ############################################################################
    ########################### Light Curves Data ##############################
    ############################################################################

    def mask_data(self, band_list=None, mask_snr=True, snr=5, mask_phase=False, min_phase=-20, max_phase=40):
        """Mask the data with the given S/N and/or within the given range of days respect to maximum in B band.

        NOTE: Bands with less or equal than 3 data points, after mask is applied, will be deleted.

        Parameters
        ----------
        band_list : list, default 'None'
            List of bands to plot. If 'None', band list is set to 'self.bands'.
        mask_snr : bool, default 'True'
            If 'True', keeps the flux values with S/N greater or equal to 'snr'.
        snr : float, default '5'
            S/N threshold applied to mask data.
        mask_phase : bool, default 'False'
            If 'True', keeps the flux values within the given phase range set by 'min_phase' and 'max_phase'.
            An initial estimation of the peak is needed first (can be set manually).
        min_phase : int, default '-20'
            Minimum phase limit applied to mask data.
        max_phase : int, default '40'
            Maximum phase limit applied to mask data.
        """

        if band_list is None:
            band_list = self.bands

        bands2delete = []

        if mask_phase:
            #assert self.tmax, 'An initial estimation of the peak is needed first!'
            if self.tmax:
                tmax = self.tmax
            else:
                self.calc_pivot()
                id_peak = np.argmax(self.data[self.pivot_band]['flux'])
                tmax = self.data[self.pivot_band]['mjd'][id_peak]

            for band in band_list:
                mask = np.where((self.data[band]['mjd'] - tmax >= min_phase*(1+self.z)) &
                                (self.data[band]['mjd'] - tmax <= max_phase*(1+self.z))
                               )
                self.data[band]['mjd'] = self.data[band]['mjd'][mask]
                self.data[band]['flux'] = self.data[band]['flux'][mask]
                self.data[band]['flux_err'] = self.data[band]['flux_err'][mask]

                if len(self.data[band]['flux']) <= 3:
                    bands2delete.append(band)

        if mask_snr:
            for band in band_list:
                mask = np.abs(self.data[band]['flux']/self.data[band]['flux_err']) >= snr
                self.data[band]['mjd'] = self.data[band]['mjd'][mask]
                self.data[band]['flux'] = self.data[band]['flux'][mask]
                self.data[band]['flux_err'] = self.data[band]['flux_err'][mask]

                if len(self.data[band]['flux']) <= 3:
                    bands2delete.append(band)

        self.delete_bands(bands2delete)  # delete bands with less than 3 data points after applying mask
        assert len(self.bands) > 1, 'The SN has not enough data (either one or no bands) left after the mask was applied.'


    def plot_data(self, band_list=None, plot_type='flux', save=False, fig_name=None, outformat='png'):
        """Plot the SN light curves.

        Negative fluxes are masked out if magnitudes are plotted.

        Parameters
        ----------
        band_list : list, default None
            List of bands to plot. If None, band list is set to 'self.bands'.
        plot_type : str, default 'flux'
            Type of value used for the data: either 'mag' or 'flux'.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default None
            Name of the saved plot. If None is used the name of the file will be '{self.name}_lcs.{outformat}'.
            Only works if 'save' is set to 'True'.
        outformat : str, default 'png'
            Output file format.

        """

        assert (plot_type=='mag' or plot_type=='flux'), f'"{plot_type}" is not a valid plot type.'
        new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]

        if band_list is None:
            band_list = self.bands

        exp = np.round(np.log10(self.data[band_list[0]]['flux'].mean()), 0)
        y_norm = 10**exp

        # to set plot limits
        if plot_type=='flux':
            plot_lim_vals = [[self.data[band]['flux'].min()/y_norm, self.data[band]['flux'].max()/y_norm] for band in self.bands]
            plot_lim_vals = np.ndarray.flatten(np.array(plot_lim_vals))
            ymin_lim = np.r_[plot_lim_vals, 0.0].min()*0.9
            if ymin_lim < 0.0:
                ymin_lim *= 1.1/0.9
            ymax_lim = plot_lim_vals.max()*1.1
        elif plot_type=='mag':
            plot_lim_vals = [[np.nanmin(flux2mag(np.abs(self.data[band]['flux']), self.data[band]['zp'])),
                              np.nanmax(flux2mag(np.abs(self.data[band]['flux']), self.data[band]['zp']))] for band in self.bands]
            plot_lim_vals = np.ndarray.flatten(np.array(plot_lim_vals))
            ymin_lim = np.nanmin(plot_lim_vals)*0.98
            ymax_lim = np.nanmax(plot_lim_vals)*1.02

        fig, ax = plt.subplots(figsize=(8,6))
        for i, band in enumerate(band_list):
            if plot_type=='flux':
                time, flux, err = np.copy(self.data[band]['mjd']), np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])
                flux, err = flux/y_norm, err/y_norm
                ax.errorbar(time, flux, err, fmt='o', mec='k', capsize=3, capthick=2, ms=8, elinewidth=3, label=band, color=new_palette[i])
                ylabel = r'Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp
            elif plot_type=='mag':
                ylabel = 'Apparent Magnitude'
                mask = np.where(self.data[band]['flux'] > 0)
                mjd = self.data[band]['mjd'][mask]
                mag, err = flux2mag(self.data[band]['flux'][mask], self.data[band]['zp'], self.data[band]['flux_err'][mask])

                ax.errorbar(mjd, mag, err, fmt='o', mec='k', capsize=3, capthick=2, ms=8, elinewidth=3, label=band, color=new_palette[i])

        ax.set_ylabel(ylabel, fontsize=16, family='serif')
        ax.set_xlabel('Modified Julian Date', fontsize=16, family='serif')
        ax.set_title(f'{self.name}\nz = {self.z:.5}', fontsize=18, family='serif')
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=16)
        ax.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True, labelsize=16)
        ax.legend(fontsize=13)
        ax.set_ylim(ymin_lim, ymax_lim)

        if plot_type=='mag':
            plt.gca().invert_yaxis()

        if save:
            if fig_name is None:
                fig_name = f'{self.name}_lcs.{outformat}'
            fig.tight_layout()
            plt.savefig(f'plots/{fig_name}')

        plt.show()


    def normalize_data(self, offsets_file=None):
        """Normalize the fluxes and zero-points (ZPs).

        Fluxes are converted to physical units by calculating the ZPs according to the
        magnitude system, either AB, BD17 or Vega.

        Parameters
        ----------
        offsets_file : str, default None
            File to use for the bands calibration of the zero point. If None, the default file is used (either
            'bd17_sys_zps.dat' or 'ab_sys_zps.dat', depending on the band).
        """

        for band in self.bands:
            mag_sys = self.data[band]['mag_sys']
            current_zp = self.data[band]['zp']

            new_zp = calc_zp(self.filters[band]['wave'], self.filters[band]['transmission'],
                                        self.filters[band]['response_type'], mag_sys, band, offsets_file)

            self.data[band]['flux'] = self.data[band]['flux']*10**(-0.4*(current_zp - new_zp))
            self.data[band]['flux_err'] = self.data[band]['flux_err']*10**(-0.4*(current_zp - new_zp))
            self.data[band]['zp'] = new_zp

    ############################################################################
    ############################ Light Curves Fits #############################
    ############################################################################

    def fit_lcs(self, kernel='matern52', kernel2='squaredexp', gp_mean='mean', fit_2d=True, fit_mag=True, use_mcmc=False):
        """Fits the data for each band using gaussian process

        The fits are done independently for each band. The initial B-band peak time is estimated with
        the pivot band as long as a peak can me calculated, having a derivative equal to zero.

        Parameters
        ----------
        kernel : str, default 'matern52'
            Kernel to be used to fit the light curves with gaussian process. E.g., 'matern52', 'matern32', 'squaredexp'.
        kernel2 : str, default 'squaredexp'
            Kernel to be used in the wavelength axis when fitting in 2D with gaussian process. E.g., 'matern52', 'matern32', 'squaredexp'.
        gp_mean : str, default 'mean'
            Mean function to be used when fitting in 1D with gaussian process. The default uses a constant function
            equal to the mean flux. Possible choices are: 'mean', 'gaussian', 'bazin', 'zheng'. 'bazin' implements the model from
            Bazin et al. (2011) while 'zheng' implements the model from Zheng et al. (2018).
        fit_2d : bool, default 'True'
            Wether to fit in 1D or 2D with gaussian process.
        fit_mag : bool, default 'True'
            If 'True' and fit_2d=True, the data is fit in magnitude space. This is recommended for 2D fit.
        use_mcmc : bool, default 'False'
            If 'True', MCMC is used for the 2D gaussian process fit, otherwise, a simple minimization with scipy.optimize is used.
        """
        ########################
        ####### GP Fit #########
        ########################

        self.calc_pivot()

        if not fit_2d:
            for band in self.bands:
                time, flux, std = gp_lc_fit(self.data[band]['mjd'], self.data[band]['flux'],
                                         self.data[band]['flux_err'], kernel=kernel, gp_mean=gp_mean)

                try:
                    peakidxs = peak.indexes(flux, thres=.3, min_dist=5//(time[1]-time[0]))
                    idx_max = np.array([idx for idx in peakidxs if all(flux[:idx]<flux[idx])]).min()
                    if flux[idx_max] == np.max(flux) and flux[idx_max] > 0:
                        tmax = time[idx_max]
                        mmax = -2.5*np.log10(flux[idx_max]) + self.data[band]['zp']
                    else:
                        tmax = mmax = np.nan
                except:
                    tmax = mmax = np.nan
                self.lc_fits[band] = {'mjd':time, 'flux':flux, 'std':std, 'tmax':tmax, 'mmax':mmax}

            tmax0 = self.lc_fits[self.pivot_band]['tmax']  # initil estimation of the peak

            bands = self.bands.copy()
            while np.isnan(tmax0):
                bands.remove(self.pivot_band)
                # check another band to work as pivot bands
                self.calc_pivot(bands)
                tmax0 = self.lc_fits[self.pivot_band]['tmax']
                if len(bands)<=2:
                    break

            assert not np.isnan(tmax0), f'Unable to obtain B-band peak for {self.name}!'
            self.tmax = self.tmax0 = np.round(tmax0, 2)

            delta_eff0 = self.filters[self.pivot_band]['eff_wave']/(1+self.z) - self.filters['Bessell_B']['eff_wave']

            # find the second closest band to restframe B-band for a more accurate tmax estimation
            eff_wave_diff = [self.filters[band]['eff_wave']/(1+self.z) - self.filters['Bessell_B']['eff_wave'] for band in self.bands]
            pivot_index = self.bands.index(self.pivot_band)
            eff_wave_diff[pivot_index] = 1e6  # "remove" the pivot band to find the 2nd closest band

            next_band_ind = np.argmin(np.array(eff_wave_diff))
            next_band = self.bands[next_band_ind]
            tmax1 = self.lc_fits[next_band]['tmax']

            if not np.isnan(tmax1):
                # estimate weighted average of tmax from two bands
                w0 = 1/delta_eff0**2
                delta_eff1 = eff_wave_diff[next_band_ind]
                w1 = 1/delta_eff1**2
                wtmax = (tmax0*w0 + tmax1*w1)/(w0 + w1)  # weighted mean
                self.tmax = self.tmax0 = np.round(wtmax, 2)

            for band in self.bands:
                self.lc_fits[band]['phase'] = (self.lc_fits[band]['mjd']-self.tmax) / (1+self.z)

        else:
            flux_array = np.hstack([self.data[band]['flux'] for band in self.bands])
            flux_err_array = np.hstack([self.data[band]['flux_err'] for band in self.bands])

            time_array = np.hstack([self.data[band]['mjd'] for band in self.bands])
            wave_array = np.hstack([[self.filters[band]['eff_wave']]*len(self.data[band]['mjd']) for band in self.bands])

            bands_waves = np.hstack([self.filters[band]['wave'] for band in self.bands])
            bands_edges = np.array([bands_waves.min(), bands_waves.max()])

            if fit_mag:
                mask = flux_array >= 0.0
                mag_array, mag_err_array = flux2mag(flux_array[mask], 0.0, flux_err_array[mask])  # ZPs are not necessary for fitting
                time_array = time_array[mask]
                wave_array = wave_array[mask]

                timeXwave, mu, std = gp_2d_fit(time_array, wave_array, mag_array, mag_err_array,
                                                kernel1=kernel, kernel2=kernel2, x2_edges=bands_edges, use_mcmc=use_mcmc)
                mu, std = mag2flux(mu, 0.0, std)

            else:
                timeXwave, mu, std = gp_2d_fit(time_array, wave_array, flux_array, flux_err_array,
                                                kernel1=kernel, kernel2=kernel2, x2_edges=bands_edges, use_mcmc=use_mcmc)

            self.lc_fits['timeXwave'], self.lc_fits['mu'], self.lc_fits['std'] = timeXwave, mu, std

            ########################
            ##### Caculate Peak ####
            ########################
            wave_ind = np.argmin(np.abs(self.filters['Bessell_B']['eff_wave']*(1+self.z) - timeXwave.T[1]))
            eff_wave = timeXwave.T[1][wave_ind]  # closest wavelength from the gp grid to the effective_wavelength*(1+z) of Bessell_B
            inds = [i for i, txw_tuplet in enumerate(timeXwave) if txw_tuplet[1]==eff_wave]

            time, flux, err = timeXwave.T[0][inds], mu[inds], std[inds]

            try:
                peakidxs = peak.indexes(flux, thres=.3, min_dist=5//(time[1]-time[0]))
                # pick the index of the first peak in case some band has 2 peaks (like IR bands)
                idx_max = np.array([idx for idx in peakidxs if all(flux[:idx]<flux[idx])]).min()
                self.tmax = self.tmax0 = np.round(time[idx_max], 2)

                phaseXwave = np.copy(timeXwave)
                phaseXwave.T[0] = (timeXwave.T[0] - self.tmax)/(1 + self.z)
                self.lc_fits['phaseXwave'] = phaseXwave
            except:
                raise ValueError(f'Unable to obtain an accurate B-band peak for {self.name}!')

            ###############################
            ## Interpolated light curves ##
            ###############################
            for band in self.bands:
                wave_ind = np.argmin(np.abs(self.filters[band]['eff_wave'] - timeXwave.T[1]))
                eff_wave = timeXwave.T[1][wave_ind]  # closest wavelength from the gp grid to the effective wavelength of the band
                inds = [i for i, txw_tuplet in enumerate(timeXwave) if txw_tuplet[1]==eff_wave]

                time, phase, flux, err = timeXwave.T[0][inds], phaseXwave.T[0][inds], mu[inds], std[inds]
                self.lc_fits[band] = {'mjd':time, 'phase':phase, 'flux':flux, 'std':err}

                # calculate observed peak for each band
                try:
                    peakidxs = peak.indexes(flux, thres=.3, min_dist=5//(time[1]-time[0]))
                    idx_max = np.array([idx for idx in peakidxs if all(flux[:idx]<flux[idx])]).min()
                    self.lc_fits[band]['tmax'] = np.round(time[idx_max], 2)
                    self.lc_fits[band]['mmax'] = -2.5*np.log10(flux[idx_max]) + self.data[band]['zp']
                except:
                    self.lc_fits[band]['tmax'] = self.lc_fits[band]['mmax'] = np.nan

    def plot_fits(self, plot_together=True, plot_type='flux', save=False, fig_name=None, outformat='png'):
        """Plots the light-curves fits results.

        Plots the observed data for each band together with the gaussian process fits. The initial B-band
        peak estimation is plotted. The final B-band peak estimation after light-curves corrections is
        also potted if corrections have been applied.

        Parameters
        ----------
        plot_together : bool, default 'True'
            If 'True', plots the bands together in one plot. Otherwise, each band is plotted separately.
        plot_type : str, default 'flux'
            Type of value used for the data: either 'mag' or 'flux'.
        save : bool, default 'False'
            If 'True', saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be '{self.name}_lc_fits.{outformat}'.
            Only works if 'save' is set to 'True'.
        outformat : str, default 'png'
            Output file format.

        """

        if plot_together:

            new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]

            exp = np.round(np.log10(np.abs(self.data[self.bands[0]]['flux']).mean()), 0)
            y_norm = 10**exp

            # to set plot limits
            if plot_type=='flux':
                plot_lim_vals = [[self.data[band]['flux'].min(), self.data[band]['flux'].max()] for band in self.bands]
                plot_lim_vals = np.ndarray.flatten(np.array(plot_lim_vals))/y_norm
                ymin_lim = np.r_[plot_lim_vals, 0.0].min()*0.9
                if ymin_lim < 0.0:
                    ymin_lim *= 1.1/0.9
                ymax_lim = plot_lim_vals.max()*1.05
            elif plot_type=='mag':
                plot_lim_vals = [[np.nanmin(flux2mag(np.abs(self.data[band]['flux']), self.data[band]['zp'])),
                                  np.nanmax(flux2mag(np.abs(self.data[band]['flux']), self.data[band]['zp']))] for band in self.bands]
                plot_lim_vals = np.ndarray.flatten(np.array(plot_lim_vals))
                ymin_lim = np.nanmin(plot_lim_vals)*0.98
                ymax_lim = np.nanmax(plot_lim_vals)*1.02

            fig, ax = plt.subplots(figsize=(8, 6))
            for i, band in enumerate(self.bands):

                time, flux, std = np.copy(self.lc_fits[band]['mjd']), np.copy(self.lc_fits[band]['flux']), np.copy(self.lc_fits[band]['std'])
                data_time, data_flux, data_std = np.copy(self.data[band]['mjd']), np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])

                if plot_type=='flux':
                    flux, std = flux/y_norm, std/y_norm
                    data_flux, data_std = data_flux/y_norm, data_std/y_norm

                    ax.errorbar(data_time, data_flux, data_std, fmt='o', mec='k', capsize=3, capthick=2, ms=8,
                                    elinewidth=3, color=new_palette[i],label=band)
                    ax.plot(time, flux,'-', color=new_palette[i], lw=2, zorder=16)
                    ax.fill_between(time, flux-std, flux+std, alpha=0.5, color=new_palette[i])
                    ax.set_ylabel(r'Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp, fontsize=16, family='serif')
                    #ax.set_ylabel(r'Scaled Flux', fontsize=16, family='serif')

                elif plot_type=='mag':
                    # avoid non-positive numbers in logarithm
                    fit_mask = flux > 0
                    time, flux, std = time[fit_mask], flux[fit_mask], std[fit_mask]
                    data_mask = data_flux > 0
                    data_time, data_flux, data_std = data_time[data_mask], data_flux[data_mask], data_std[data_mask]

                    mag, err = flux2mag(flux, self.data[band]['zp'], std)
                    data_mag, data_err = flux2mag(data_flux, self.data[band]['zp'], data_std)

                    ax.errorbar(data_time, data_mag, data_err, fmt='o', mec='k', capsize=3, capthick=2, ms=8,
                                elinewidth=3, color=new_palette[i],label=band)
                    ax.plot(time, mag,'-', color=new_palette[i], lw=2, zorder=16)
                    ax.fill_between(time, mag-err, mag+err, alpha=0.5, color=new_palette[i])
                    ax.set_ylabel(r'Apparent Magnitude [mag]', fontsize=16, family='serif')

            ax.axvline(x=self.tmax0, color='k', linestyle='--', alpha=0.4)
            ax.axvline(x=self.tmax, color='k', linestyle='--')
            ax.minorticks_on()
            ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True, labelsize=16)
            ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True, labelsize=16)
            ax.set_xlabel('Modified Julian Date', fontsize=16, family='serif')

            ax.set_title(f'{self.name}\nz = {self.z:.5}', fontsize=18, family='serif')
            ax.legend(fontsize=13, loc='upper right')
            ax.set_ylim(ymin_lim, ymax_lim)

            if plot_type=='mag':
                plt.gca().invert_yaxis()
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
                ax.errorbar(self.data[band]['mjd'], self.data[band]['flux'], self.data[band]['flux_err'], fmt='og',
                                capsize=3, capthick=2, ms=8, elinewidth=3, mec='k')
                ax.plot(time, flux,'-', lw=2, zorder=16, color='k')
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
            #fig.text(0.04, 0.5, r'Scaled Flux', va='center', rotation='vertical', fontsize=18, family='serif')

        if save:
            if fig_name is None:
                fig_name = f'{self.name}_lc_fits.{outformat}'
            fig.tight_layout()
            plt.savefig(f'plots/{fig_name}')

        plt.show()

    ############################################################################
    ######################### Light Curves Correction ##########################
    ############################################################################

    def mangle_sed(self, min_phase=-15, max_phase=30, kernel='squaredexp', gp_mean='mean', correct_extinction=True):
        """Mangles the SED with the given method to match the SN magnitudes.

        Parameters
        ----------
        min_phase : int, default '-15'
            Minimum phase to mangle.
        max_phase : int, default '30'
            Maximum phase to mangle.
        kernel : str, default 'squaredexp'
            Kernel to be used for the gaussian process fit of the mangling function.  E.g, 'matern52',
            'matern32', 'squaredexp'.
        gp_mean: str, default 'mean'
            Mean function to be used when fitting with gaussian process. The default uses a constant function
            equal to the mean of the values. Possible choices are: 'mean', 'poly'. 'poly' uses a 3rd degree polynomial function.
        correct_extinction: bool, default 'True'
            Whether or not to correct for Milky Way extinction.
        """

        phases = np.arange(min_phase, max_phase+1, 1)

        self.user_input['mangle_sed'] = {'min_phase':min_phase, 'max_phase':max_phase,
                                            'kernel':kernel, 'correct_extinction':correct_extinction}
        lc_phases = self.lc_fits[self.pivot_band]['phase']

        ####################################
        ##### Calculate SED photometry #####
        ####################################
        sed_df = self.sed['info'].copy()
        sed_df = sed_df[(lc_phases.min() <= sed_df.phase) & (sed_df.phase <= lc_phases.max())]  # to match the available epochs from the lcs
        sed_df = sed_df[sed_df.phase.isin(phases)]  # to match the requested epochs

        # first redshift the SED ("move" it in z) and then apply extinction from MW only
        sed_df.wave, sed_df.flux = sed_df.wave.values*(1+self.z), sed_df.flux.values/(1+self.z)
        if correct_extinction:
            sed_df.flux = redden(sed_df.wave.values, sed_df.flux.values, self.ra, self.dec)

        self.sed_lcs = {band:{'flux':[], 'mjd':None, 'phase':None} for band in self.bands}
        sed_phases = sed_df.phase.unique()

        # calculate SED light curves
        for phase in sed_phases:
            phase_df = sed_df[sed_df.phase==phase]
            for band in self.bands:
                band_flux = integrate_filter(phase_df.wave.values, phase_df.flux.values, self.filters[band]['wave'],
                                       self.filters[band]['transmission'], self.filters[band]['response_type'])
                self.sed_lcs[band]['flux'].append(band_flux)

        for band in self.bands:
            self.sed_lcs[band]['flux'] = np.array(self.sed_lcs[band]['flux'])
            self.sed_lcs[band]['phase'] = sed_phases
            self.sed_lcs[band]['mjd'] = sed_phases*(1+self.z) + self.tmax

        ###################################
        ####### set-up for mangling #######
        ###################################
        # find the fluxes at the exact SED phases
        obs_flux_dict = {band:np.interp(sed_phases, self.lc_fits[band]['phase'], self.lc_fits[band]['flux']) for band in self.bands}
        obs_err_dict = {band:np.interp(sed_phases, self.lc_fits[band]['phase'], self.lc_fits[band]['std']) for band in self.bands}
        flux_ratios_dict = {band:obs_flux_dict[band]/self.sed_lcs[band]['flux'] for band in self.bands}

        wave_array = np.array([self.filters[band]['eff_wave'] for band in self.bands])
        bands_waves = np.hstack([self.filters[band]['wave'] for band in self.bands])
        x_edges = np.array([bands_waves.min(), bands_waves.max()])  # to includes the edges of the reddest and bluest bands

        ################################
        ########## mangle SED ##########
        ################################
        self.mangled_sed = pd.DataFrame(columns=['phase', 'wave', 'flux', 'err'])
        for i, phase in enumerate(sed_phases):
            obs_fluxes = np.array([obs_flux_dict[band][i] for band in self.bands])
            obs_errs = np.array([obs_err_dict[band][i] for band in self.bands])
            flux_ratios_array = np.array([flux_ratios_dict[band][i] for band in self.bands])

            phase_df = sed_df[sed_df.phase==phase]
            sed_epoch_wave, sed_epoch_flux = phase_df.wave.values, phase_df.flux.values

            # mangling routine including optimisation
            mangling_results = mangle(wave_array, flux_ratios_array, sed_epoch_wave, sed_epoch_flux,
                                        obs_fluxes, obs_errs, self.bands, self.filters, kernel, gp_mean, x_edges)

            # precision of the mangling function
            mag_diffs = {band:-2.5*np.log10(mangling_results['flux_ratios'][i]) if mangling_results['flux_ratios'][i] > 0
                                                else np.nan for i, band in enumerate(self.bands)}
            self.mangling_results.update({phase:mangling_results})
            self.mangling_results[phase].update({'mag_diff':mag_diffs})

            # save the SED phase info into a DataFrame
            mangled_sed = mangling_results['mangled_sed']
            mangled_wave, mangled_flux, mangled_flux_err = mangled_sed['wave'], mangled_sed['flux'], mangled_sed['flux_err']
            phase_info = np.array([[phase]*len(mangled_wave), mangled_wave, mangled_flux, mangled_flux_err]).T
            phase_df = pd.DataFrame(data=phase_info, columns=['phase', 'wave', 'flux', 'err'])
            self.mangled_sed = pd.concat([self.mangled_sed, phase_df])

        # correct mangled SED for MW extinction first and then de-redshift it ("move" it back in z)
        self.corrected_sed = self.mangled_sed.copy()
        if correct_extinction:
            self.corrected_sed.flux = deredden(self.corrected_sed.wave.values, self.corrected_sed.flux.values, self.ra, self.dec)
        self.corrected_sed.wave = self.corrected_sed.wave.values/(1+self.z)
        self.corrected_sed.flux = self.corrected_sed.flux.values*(1+self.z)


    def plot_mangling_function(self, phase=0, mangling_function_only=False, verbose=True, save=False, fig_name=None, outformat='png'):
        """Plot the mangling function for a given phase.

        Parameters
        ----------
        phase : int, default '0'
            Phase to plot the mangling function. By default it plots the mangling function at B-band peak.
        mangle_only : bool, default 'False'
            If 'True', only plots the mangling function, else, plots the SEDs and filters as well (with scaled values).
        verbose : bool, default 'True'
            If 'True', returns the difference between the magnitudes from the fits and the magnitudes from the
            modified SED after mangling, for each of the bands in 'band_list'.
        save : bool, default 'False'
            If true, saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be '{self.name}_mangling_phase{phase}.{outformat}'.
            Only works if 'save' is set to 'True'.
        outformat : str, default 'png'
            Output file format.

        """

        assert (phase in self.mangling_results.keys()), f'phase {phase} does not have a mangling result.'

        man = self.mangling_results[phase]
        eff_waves = np.copy(man['init_flux_ratios']['waves'])
        init_flux_ratios = np.copy(man['init_flux_ratios']['flux_ratios'])
        flux_ratios_err = np.copy(man['init_flux_ratios']['flux_ratios_err'])

        opt_flux_ratios = np.copy(man['opt_flux_ratios']['flux_ratios'])
        obs_fluxes = np.copy(man['obs_band_fluxes']['fluxes'])
        sed_fluxes = np.copy(man['sed_band_fluxes']['fluxes'])

        x, y, yerr = np.copy(man['mangling_function']['waves']), np.copy(man['mangling_function']['flux_ratios']), np.copy(man['mangling_function']['flux_ratios_err'])
        mang_sed_wave, mang_sed_flux = man['mangled_sed']['wave'], man['mangled_sed']['flux']
        init_sed_wave, init_sed_flux = man['init_sed']['wave'], man['init_sed']['flux']

        kernel = man['kernel']
        bands = man['mag_diff'].keys()

        if mangling_function_only:
            fig, ax = plt.subplots(figsize=(8,6))
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
            ax.set_ylabel(r'(Flux$_{\rm Obs}$ / Flux$_{\rm Temp}) \times$ 10$^{%.0f}$'%exp, fontsize=16, family='serif')
            ax.minorticks_on()
            ax.tick_params(which='both', length=8, width=1, direction='in', right=True, labelsize=16)
            ax.tick_params(which='minor', length=4)
            ax.set_ylim(y.min()*0.95, y.max()*1.03)

            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticklabels((ax.get_xticks()/(1+self.z)).astype(int))
            ax2.minorticks_on()
            ax2.set_xlabel(r'Rest-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax2.tick_params(which='both', length=8, width=1, direction='in', labelsize=16)
            ax2.tick_params(which='minor', length=4)

            for i, band in enumerate(bands):
                x1, y1 = ax.transLimits.transform((eff_waves[i], init_flux_ratios[i]))
                ax.text(x1, y1+(-1)**i*0.12, band, horizontalalignment='center',
                        verticalalignment='center', transform=ax.transAxes, fontsize=14)
            ax.legend(loc='upper right', fontsize=12)

        else:
            fig, ax = plt.subplots(figsize=(8,6))
            ax2 = ax.twiny()
            ax3 = ax.twinx()

            norm = 2  # for bands
            norm2 = 1  # for SEDs
            index = (len(bands)-1)//2  # index of the band to do relative comparison

            init_norm = np.sum(init_sed_wave*init_sed_flux)/np.sum(init_sed_wave)
            init_sed_flux2 = init_sed_flux/init_norm
            sed_fluxes2 =sed_fluxes/init_norm

            obs_norm = np.sum(mang_sed_wave*mang_sed_flux)/np.sum(mang_sed_wave)
            mang_sed_flux2 = mang_sed_flux/obs_norm
            obs_fluxes2 = obs_fluxes/obs_norm

            bands_norm = init_sed_flux2.max()

            # filters
            for i, band in enumerate(bands):
                wave, trans = self.filters[band]['wave'], self.filters[band]['transmission']
                ax3.plot(wave, trans/trans.max()*bands_norm, color='k', alpha=0.4)

            # mangling function
            ax.plot(x, y/(obs_norm/init_norm), 'green')
            ax.fill_between(x, (y-yerr)/(obs_norm/init_norm), (y+yerr)/(obs_norm/init_norm), alpha=0.2, color='green')
            indexes = [np.argmin(np.abs(x-wave_val)) for wave_val in eff_waves]
            ax.plot(eff_waves, y[indexes]/(obs_norm/init_norm), 'sg', ms=8, mec='k')

            # initial sed and fluxes
            ax3.plot(init_sed_wave, init_sed_flux2, '--k', lw=3)  # initial sed
            ax3.plot(eff_waves, sed_fluxes2, 'ok', ms=14, label='Initial SED values', alpha=0.8, fillstyle='none')  # initial sed fluxes

            # optimized sed and fluxes
            ax3.plot(mang_sed_wave, mang_sed_flux2, 'red', lw=3)  # mangled sed
            ax3.plot(eff_waves, obs_fluxes2,'*r', ms=14, mec='k', label='Mangled SED values')  # optimized fluxes

            ax.set_xlabel(r'Observer-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax.set_ylabel(r'Scaled Mangling Function', fontsize=16, family='serif', color='g')
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim((y/(obs_norm/init_norm)).min()*0.8, (y/(obs_norm/init_norm)).max()*1.2)
            ax.tick_params(which='both', length=8, width=1, direction='in', labelsize=16)
            ax.tick_params(which='minor', length=4)
            ax.tick_params(axis='y', which='both', colors='g')
            ax.spines['left'].set_color('g')

            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticklabels((ax.get_xticks()/(1+self.z)).astype(int))
            ax2.set_xlabel(r'Rest-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax2.minorticks_on()
            ax2.tick_params(which='both', length=8, width=1, direction='in', labelsize=16)
            ax2.tick_params(which='minor', length=4)

            ax3.set_ylim(0, None)
            ax3.set_ylabel(r'Scaled Flux', fontsize=16, family='serif', rotation=270, labelpad=20)
            ax3.minorticks_on()
            ax3.tick_params(which='both', length=8, width=1, direction='in', labelsize=16)
            ax3.tick_params(which='minor', length=4)
            ax3.legend(loc='upper right', fontsize=12)

        if save:
            if fig_name is None:
                fig_name = f'{self.name}_mangling_phase{phase}.{outformat}'
            fig.tight_layout()
            plt.savefig(f'plots/{fig_name}')

        plt.show()

        if verbose:
            print(f'Mangling results, i.e., difference between mangled SED and "observed" magnitudes, at phase {phase}:')
            for band, diff in man['mag_diff'].items():
                print(f'{band}: {np.round(diff, 4):.4f} [mags]')


    def _calculate_corrected_lcs(self):
        """Calculates the SN light curves applying extinction and k-corrections.

        Note: this function is used inside self.calculate_lc_params()
        """

        corrected_lcs = {}
        lcs_fluxes = []
        lcs_errs = []
        phases = self.corrected_sed.phase.unique()

        for phase in phases:
            phase_df = self.corrected_sed[self.corrected_sed.phase==phase]
            phase_wave = phase_df.wave.values
            phase_flux = phase_df.flux.values
            phase_err = phase_df.err.values

            bands_flux = []
            bands_err = []
            for band in self.filters.keys():
                try:
                    bands_flux.append(integrate_filter(phase_wave, phase_flux, self.filters[band]['wave'], self.filters[band]['transmission'],
                                                                    self.filters[band]['response_type']))
                    bands_err.append(integrate_filter(phase_wave, phase_err, self.filters[band]['wave'], self.filters[band]['transmission'],
                                                                    self.filters[band]['response_type']))
                except:
                    bands_flux.append(np.nan)
                    bands_err.append(np.nan)

            lcs_fluxes.append(bands_flux)
            lcs_errs.append(bands_err)
        lcs_fluxes = np.array(lcs_fluxes)
        lcs_errs = np.array(lcs_errs)
        corrected_lcs = {band:{'phase':phases, 'flux':lcs_fluxes.T[i], 'err':lcs_errs.T[i]} for i, band in enumerate(self.filters.keys())}
        self.corrected_lcs = corrected_lcs

        # simple, independent 1D fit to the corrected light curves
        corrected_lcs_fit = {band:None for band in self.filters.keys()}
        for band in self.filters.keys():
            try:
                phases, fluxes, errs = corrected_lcs[band]['phase'], corrected_lcs[band]['flux'], corrected_lcs[band]['err']
                phases_fit, fluxes_fit, _ = gp_lc_fit(phases, fluxes, fluxes*1e-3)
                errs_fit = np.interp(phases_fit, phases, errs)  # linear extrapolation of errors
                corrected_lcs_fit[band] = {'phase':phases_fit, 'flux':fluxes_fit, 'err':errs_fit}
            except:
                corrected_lcs_fit[band] = {'phase':np.nan, 'flux':np.nan, 'err':np.nan}
        self.corrected_lcs_fit = corrected_lcs_fit


    def calculate_lc_params(self, maxiter=5):
        """Calculates the light-curves parameters.

        Estimation of B-band peak apparent magnitude (mb), stretch (dm15) and color ((B-V)max) parameters.
        An interpolation of the corrected light curves is done as well as part of this process.

        Parameters
        ----------
        maxiter : int, default '5'
            Maximum number of iteration of the correction process to estimate an accurate B-band peak.

        """

        self._calculate_corrected_lcs()

        ########################################
        ########### Check B-band max ###########
        ########################################
        bmax_needs_check = True
        iter = 0
        while bmax_needs_check:
            b_data = self.corrected_lcs_fit['Bessell_B']
            b_phase, b_flux = b_data['phase'], b_data['flux']
            try:
                peakidxs = peak.indexes(b_flux, thres=.3, min_dist=5//(b_phase[1]-b_phase[0]))
                idx_max = np.array([idx for idx in peakidxs if all(b_flux[:idx]<b_flux[idx])]).min()
                tmax_offset = np.round(b_phase[idx_max] - 0.0, 2)
            except:
                tmax_offset = None

            assert tmax_offset is not None, "The peak of the rest-frame B-band light curve can not be calculated."

            # compare tmax from the corrected restframe B-band to the initial estimation
            if np.abs(tmax_offset) >= 0.2:
                # update phase of the light curves
                self.tmax = np.round(self.tmax + tmax_offset, 2)
                try:
                    self.lc_fits['phaseXwave'].T[0] -= tmax_offset
                except:
                    pass
                for band in self.bands:
                    self.lc_fits[band]['phase'] -= tmax_offset
                # re-do mangling
                self.mangle_sed(**self.user_input['mangle_sed'])
                self._calculate_corrected_lcs()
            else:
                self.tmax_offset = tmax_offset
                self.tmax_err = np.round(np.abs(tmax_offset) + 0.5, 2)  # template has 1 day "cadence", so we assume 0.5 days error
                bmax_needs_check = False

            if iter>maxiter:
                break
            iter += 1

        ########################################
        ### Calculate Light Curve Parameters ###
        ########################################
        bessell_b = 'Bessell_B'
        zp_b = calc_zp(self.filters[bessell_b]['wave'], self.filters[bessell_b]['transmission'],
                                    self.filters[bessell_b]['response_type'], 'BD17', bessell_b)

        self.corrected_lcs[bessell_b]['zp'] = zp_b

        # B-band peak apparent magnitude
        phase_b, flux_b, flux_err_b = self.corrected_lcs[bessell_b]['phase'], self.corrected_lcs[bessell_b]['flux'], self.corrected_lcs[bessell_b]['err']
        id_bmax = np.where(phase_b==0.0)[0][0]
        mb, mb_err = flux2mag(flux_b[id_bmax], zp_b, flux_err_b[id_bmax])
        mb_err += 0.005  # the last term comes from the template error in one day uncertainty

        # Stretch parameter
        try:
            id_15 = np.where(phase_b==15.0)[0][0]
            B15, B15_err = flux2mag(flux_b[id_15], zp_b, flux_err_b[id_15])
            B15_err += 0.005  # the last term comes from the template error in one day uncertainty
            dm15 = B15 - mb
            dm15err = np.sqrt(mb_err**2 + B15_err**2)
        except:
            dm15 = dm15err = np.nan

        # Colour
        try:
            bessell_v = 'Bessell_V'
            zp_v = calc_zp(self.filters[bessell_v]['wave'], self.filters[bessell_v]['transmission'],
                                        self.filters[bessell_v]['response_type'], 'BD17', bessell_v)

            self.corrected_lcs[bessell_v]['zp'] = zp_v
            phase_v, flux_v, flux_err_v = self.corrected_lcs[bessell_v]['phase'], self.corrected_lcs[bessell_v]['flux'], self.corrected_lcs[bessell_v]['err']

            id_v0 = np.where(phase_v==0.0)[0][0]
            V0, V0_err = flux2mag(flux_v[id_v0], zp_v, flux_err_v[id_v0])
            V0_err += 0.011  # the last term comes from the template error in one day uncertainty
            color = mb - V0
            dcolor = np.sqrt(mb_err**2 + V0_err**2)
        except:
            color = dcolor = np.nan

        self.lc_parameters = {'mb':mb, 'dmb':mb_err, 'dm15':dm15,
                              'dm15err':dm15err, 'color':color, 'dcolor':dcolor}


    def display_results(self, band='Bessell_B', plot_type='mag', display_params=False, save=False, fig_name=None, outformat='png'):
        """Displays the rest-frame light curve for the given band.

        Plots the rest-frame band light curve together with a gaussian fit to it. The parameters estimated with
        'calculate_lc_params()' are shown as well.

        Parameters
        ----------
        band : str, default 'Bessell_B'
            Name of the band to be plotted.
        plot_type : str, default 'mag'
            Type of value used for the data: either 'mag' or 'flux'.
        display_params : bool, default 'False'
            If 'True', the light-curves parameters are displayed in the plot.
        save : bool, default 'False'
            If 'True', saves the plot into a file.
        fig_name : str, default 'None'
            Name of the saved plot. If 'None' is used the name of the file will be '{self.name}_restframe_{band}.{outformat}'.
            Only works if 'save' is set to 'True'.
        outformat : str, default 'png'
            Output file format.

        """

        assert (plot_type=='mag' or plot_type=='flux'), f'"{plot_type}" is not a valid plot type.'

        mb = self.lc_parameters['mb']
        dmb = self.lc_parameters['dmb']
        color = self.lc_parameters['color']
        dcolor = self.lc_parameters['dcolor']
        dm15 = self.lc_parameters['dm15']
        dm15err = self.lc_parameters['dm15err']

        if band is None:
            band = 'Bessell_B'

        x = np.copy(self.corrected_lcs[band]['phase'])
        y = np.copy(self.corrected_lcs[band]['flux'])
        yerr = np.copy(self.corrected_lcs[band]['err'])
        zp = self.corrected_lcs[band]['zp']
        x_fit = np.copy(self.corrected_lcs_fit[band]['phase'])
        y_fit = np.copy(self.corrected_lcs_fit[band]['flux'])
        yerr_fit = np.copy(self.corrected_lcs_fit[band]['err'])

        if plot_type=='flux':
            exp = np.round(np.log10(y.max()), 0)
            y_norm = 10**exp
            y /= y_norm
            yerr /= y_norm
            y_fit /= y_norm
            yerr_fit /= y_norm

        elif plot_type=='mag':
            # y, yerr, y_fit, yerr_fit variables get reassigned
            y, yerr = flux2mag(y, zp, yerr)
            y_fit, yerr_fit = flux2mag(y_fit, zp, yerr_fit)

        fig, ax = plt.subplots(figsize=(8,6))
        ax.errorbar(x, y, yerr, fmt='-.o', color='k', ecolor='k', mec='k', capsize=3, capthick=2, ms=8, elinewidth=3, zorder=16)
        ax.plot(x_fit, y_fit, 'c-', alpha=0.7)
        ax.fill_between(x_fit, y_fit+yerr_fit, y_fit-yerr_fit, alpha=0.5, color='c')

        if display_params:
            ax.text(0.75, 0.9,r'm$_B^{\rm max}$=%.3f$\pm$%.3f'%(mb, dmb), ha='center', va='center', fontsize=15, transform=ax.transAxes)
            ax.text(0.75, 0.8,r'$\Delta$m$_{15}$($B$)=%.3f$\pm$%.3f'%(dm15, dm15err), ha='center', va='center', fontsize=15, transform=ax.transAxes)
            ax.text(0.75, 0.7,r'($B-V$)$_{\rm max}$=%.3f$\pm$%.3f'%(color, dcolor), ha='center', va='center', fontsize=15, transform=ax.transAxes)

        ax.set_xlabel(f'Phase with respect to B-band peak [days]', fontsize=16, family='serif')
        ax.set_title(f'{self.name}\n{band}, z={self.z:.5}, t0={self.tmax:.2f}', fontsize=16, family='serif')
        if plot_type=='flux':
            ax.set_ylabel(r'Flux [10$^{%.0f}$ erg cm$^{-2}$ s$^{-1}$ $\AA^{-1}$]'%exp, fontsize=16, family='serif')
            ax.set_ylim(y.min()*0.90, y.max()*1.05)
        elif plot_type=='mag':
            ax.set_ylabel('Apparent Magnitude', fontsize=16, family='serif')
            ax.set_ylim(np.nanmin(y)*0.98, np.nanmax(y)*1.02)
            plt.gca().invert_yaxis()
        ax.minorticks_on()
        ax.tick_params(which='major', length=8, width=1, direction='in', top=True, right=True, labelsize=16)
        ax.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True, labelsize=16)

        if save:
            if fig_name is None:
                fig_name = f'{self.name}_restframe_{band}.{outformat}'
            fig.tight_layout()
            plt.savefig(f'plots/{fig_name}')

        plt.show()


    def do_magic(self):
        """Applies the whole correction process with default settings to obtain restframe light curves and light-curve parameters.

        Note: this is meant to be used for "quick" fits.
        """
        self.normalize_data()
        self.fit_lcs()
        self.mangle_sed()
        self.calculate_lc_params()