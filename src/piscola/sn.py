# -*- coding: utf-8 -*-
# This is the skeleton of PISCOLA, the main file

import piscola
from .filter_utils import integrate_filter, calc_eff_wave, calc_pivot_wave, calc_zp, filter_effective_range
from .gaussian_process import gp_lc_fit, gp_2d_fit
from .extinction_correction import redden, deredden, calculate_ebv
from .mangling import mangle
from .pisco_utils import trim_filters, flux2mag, mag2flux, change_zp

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from peakutils import peak
import pickle5 as pickle
import pandas as pd
import numpy as np
import random
import math
import glob
import os

### Initialisation functions ###
# These are mainly used by the 'sn' class below

def _initialise_sn(sn_file):
    """Initialise the :func:`sn` object.

    The object is initialised with all the necessary information like filters, fluxes, etc.

    Parameters
    ----------
    sn_file : str
        Name of the SN or SN file.

    Returns
    -------
    sn_obj : obj
        New :func:`sn` object.

    """

    name, z, ra, dec = pd.read_csv(sn_file, delim_whitespace=True, nrows=1,
                                    converters={'name':str, 'z':float, 'ra':float, 'dec':float}).iloc[0].values
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
        time, flux = band_info['time'].values, band_info['flux'].values
        flux_err, zp = band_info['flux_err'].values, float(band_info['zp'].unique()[0])
        mag, mag_err = flux2mag(flux, zp, flux_err)
        mag_sys = band_info['mag_sys'].unique()[0]

        sn_obj.data[band] = {'time':time,
                             'flux':flux,
                             'flux_err':flux_err,
                             'mag':mag,
                             'mag_err':mag_err,
                             'zp':zp,
                             'mag_sys':mag_sys,
                            }

    return sn_obj


def call_sn(sn_file, directory='data'):
    """Loads a supernova from a file and initialises it.

    Parameters
    ----------
    sn_file: str
        Name of the SN or SN file.
    directory : str, default ``data``
        Directory where to look for the SN file unless the full or relative path is given in ``sn_file``.

    """

    sn_full_path = os.path.join(directory, sn_file)
    # if sn_file is the file name
    if os.path.isfile(sn_full_path):
        return _initialise_sn(sn_full_path)

    # if sn_file is the SN name
    elif os.path.isfile(sn_full_path + '.dat'):
        return _initialise_sn(sn_full_path + '.dat')

    # if sn_file is the file name with full or relative path
    elif os.path.isfile(sn_file):
        return _initialise_sn(sn_file)

    else:
        raise ValueError(f'{sn_file} was not a valid SN name or file.')


def load_sn(name, path=None):
    """Loads a :func:`sn` oject that was previously saved as a pickle file.

    Parameters
    ----------
    name : str
        Name of the SN object.
    path: str, default ``None``
        Path where to save the SN file given the ``name``.

    Returns
    -------
    pickle.load(file) : obj
        :func:`sn` object previously saved as a pickle file.

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

# This is the main class
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
        name : str, default ``None``
            Name of the SN object. If no name is given, ``name`` is set to ``self.name``.
        path: str, default ``None``
            Path where to save the SN file given the ``name``.
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
        """Obtains the transmission functions for the observed filters and the Bessell filters as well.
        """

        path = piscola.__path__[0]
        sed_df = self.sed['data']
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

                    # retrieve response type; if none, assumed to be photon type
                    try:
                        with open(os.path.join(root, 'response_type.txt')) as resp_file:
                            for line in resp_file:
                                response_type = line.split()[0].lower()
                    except:
                        response_type = 'photon'
                    assert response_type in ['photon', 'energy'], f'"{response_type}" is not a valid response type \
                                                                            ("photon" or "energy") for {band} filter.'

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

            # retrieve response type; if none, assumed to be photon type
            try:
                with open(os.path.join(root, 'response_type.txt')) as resp_file:
                    for line in resp_file:
                        response_type = line.split()[0].lower()
            except:
                response_type = 'photon'
            assert response_type in ['photon', 'energy'], f'"{response_type}" is not a valid response type \
                                                                    ("photon" or "energy") for {band} filter.'

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
            List of filters.
        response_type : str, default ``photon``
            Response type of the filter. The options are: ``photon`` and ``energy``.

        """

        path = piscola.__path__[0]
        sed_df = self.sed['data']
        sed_df = sed_df[sed_df.phase==0.0]
        sed_wave, sed_flux = sed_df.wave.values, sed_df.flux.values

        if type(filter_list)==str:
            filter_list = [filter_list]

        for band in filter_list:
            file = f'{band}.dat'

            for root, dirs, files in os.walk(os.path.join(path, 'filters')):
                if file in files:
                    wave0, transmission0 = np.loadtxt(os.path.join(root, file)).T
                    # linearly interpolate filters
                    wave = np.linspace(wave0.min(), wave0.max(), int(wave0.max() - wave0.min()))
                    transmission = np.interp(wave, wave0, transmission0, left=0.0, right=0.0)
                    # remove long tails of zero values on both edges
                    imin, imax = trim_filters(transmission)
                    wave, transmission = wave[imin:imax], transmission[imin:imax]

                    # retrieve response type; if none, assumed to be photon type
                    try:
                        with open(os.path.join(root, 'response_type.txt')) as resp_file:
                            for line in resp_file:
                                response_type = line.split()[0].lower()
                    except:
                        response_type = 'photon'
                    assert response_type in ['photon',
                                             'energy'], f'"{response_type}" is not a valid response type \
                                                                             ("photon" or "energy") for {band} filter.'

                    self.filters[band] = {'wave': wave,
                                          'transmission': transmission,
                                          'eff_wave': calc_eff_wave(sed_wave, sed_flux, wave,
                                                                    transmission, response_type=response_type),
                                          'response_type': response_type}


    def plot_filters(self, filter_list=None, save=False):
        """Plot the filters' transmission functions.

        Parameters
        ----------
        filter_list : list, default ``None``
            List of bands.
        save : bool, default ``False``
            If ``True``, saves the plot into a file with the name "filters.png".

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
            #fig.tight_layout()
            plt.savefig('filters.png')

        plt.show()


    def calc_pivot(self, band_list=None):
        """Calculates the observed band closest to Bessell-B filter.

        Parameters
        ----------
        filter_list : list, default ``None``
            List of bands.

        """

        BessellB_eff_wave = self.filters['Bessell_B']['eff_wave']

        if band_list is None:
            band_list = self.bands

        bands_eff_wave =  np.array([self.filters[band]['eff_wave']/(1+self.z) for band in band_list])
        idx = (np.abs(BessellB_eff_wave - bands_eff_wave)).argmin()
        self.pivot_band = band_list[idx]


    def remove_bands(self, bands, verbose=False):
        """Remove chosen bands together with the data in it.

        Parameters
        ----------
        bands : str or list
            Band string (for a single band) or list of bands to be removed.
        verbose : bool, default ``False``
            If ``True``, a warning is given when a band from ``bands_list`` is not found within the SN bands.

        """

        if isinstance(bands, str):
            bands = [bands]

        for band in bands:
            self.data.pop(band, None)
            self.filters.pop(band, None)
            if band in self.bands:
                self.bands.remove(band)

    ############################################################################
    ############################### SED template ###############################
    ############################################################################

    def print_sed_templates(self):
        """Prints all the available SED templates in the ``templates`` directory.
        """

        path = piscola.__path__[0]
        template_path = os.path.join(path, "templates")
        print('List of available SED templates:', [name for name in os.listdir(template_path)
                                                           if os.path.isdir(os.path.join(template_path, name))])


    def set_sed_template(self, template='jla'):
        """Sets the SED template to be used for the mangling function.

        **Note:** use :func:`print_sed_templates()` to see a list of available templates.

        Parameters
        ----------
        template : str, default ``jla``
            Template name. E.g., ``jla``, ``conley09f``, etc.

        """
        # This can be modified to accept other templates
        path = piscola.__path__[0]
        file = os.path.join(path, f'templates/{template}/snflux_1a.dat')
        self.sed['data'] = pd.read_csv(file, delim_whitespace=True, names=['phase', 'wave', 'flux'])
        self.sed['name'] = template


    def set_eff_wave(self):
        """Sets the effective wavelength of each band using the current state of the SED."""

        for band in self.filters.keys():
            self.filters[band]['eff_wave'] = calc_eff_wave(self.sed['wave'],
                                                           self.sed['flux'],
                                                           self.filters[band]['wave'],
                                                           self.filters[band]['transmission'],
                                                           self.filters[band]['response_type'])

    ############################################################################
    ########################### Light Curves Data ##############################
    ############################################################################

    def mask_data(self, band_list=None, mask_snr=True, snr=5, mask_phase=False, min_phase=-20, max_phase=40):
        """Mask the data with the given signal-to-noise (S/N) in flux space and/or given range of days with respect to
        B-band peak.

        **Note:** If the light curves were not previously fitted, the phases are taken with respect to the measurement
        with the largest flux.

        Parameters
        ----------
        band_list : list, default ``None``
            List of bands to plot. If ``None``, band list is set to ``self.bands``.
        mask_snr : bool, default ``True``
            If ``True``, keeps the flux values with S/N greater or equal to ``snr``.
        snr : float, default ``5``
            S/N threshold applied to mask data in flux space.
        mask_phase : bool, default ``False``
            If ``True``, keeps the flux values within the given phase range set by ``min_phase`` and ``max_phase``.
            An initial estimation of the peak is needed first (can be set manually).
        min_phase : int, default ``-20``
            Minimum phase limit applied to mask data.
        max_phase : int, default ``40``
            Maximum phase limit applied to mask data.
        """

        if band_list is None:
            band_list = self.bands

        bands2remove = []

        if mask_phase:
            #assert self.tmax, 'An initial estimation of the peak is needed first!'
            if self.tmax:
                tmax = self.tmax
            else:
                self.calc_pivot()
                id_peak = np.argmax(self.data[self.pivot_band]['flux'])
                tmax = self.data[self.pivot_band]['time'][id_peak]

            for band in band_list:
                mask = np.where((self.data[band]['time'] - tmax >= min_phase*(1+self.z)) &
                                (self.data[band]['time'] - tmax <= max_phase*(1+self.z))
                               )
                self.data[band]['time'] = self.data[band]['time'][mask]
                self.data[band]['flux'] = self.data[band]['flux'][mask]
                self.data[band]['flux_err'] = self.data[band]['flux_err'][mask]
                self.data[band]['mag'] = self.data[band]['mag'][mask]
                self.data[band]['mag_err'] = self.data[band]['mag_err'][mask]

                if len(self.data[band]['flux']) == 0:
                    bands2remove.append(band)

        if mask_snr:
            for band in band_list:
                mask = np.abs(self.data[band]['flux']/self.data[band]['flux_err']) >= snr
                self.data[band]['time'] = self.data[band]['time'][mask]
                self.data[band]['flux'] = self.data[band]['flux'][mask]
                self.data[band]['flux_err'] = self.data[band]['flux_err'][mask]
                self.data[band]['mag'] = self.data[band]['mag'][mask]
                self.data[band]['mag_err'] = self.data[band]['mag_err'][mask]

                if len(self.data[band]['flux']) == 0:
                    bands2remove.append(band)

        self.remove_bands(bands2remove)


    def plot_data(self, band_list=None, plot_type='flux', save=False, fig_name=None):
        """Plot the SN light curves.

        Negative fluxes are masked out if magnitudes are plotted.

        Parameters
        ----------
        band_list : list, default ``None``
            List of bands to plot. If ``None``, band list is set to ``self.bands``.
        plot_type : str, default ``flux``
            Type of value used for the data: either ``mag`` or ``flux``.
        save : bool, default ``False``
            If true, saves the plot into a file.
        fig_name : str, default ``None``
            Name of the saved plot. If None is used the name of the file will be '{``self.name``}_lcs.png'.
            Only works if ``save`` is set to ``True``.

        """

        assert (plot_type=='mag' or plot_type=='flux'), f'"{plot_type}" is not a valid plot type.'
        new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]

        if band_list is None:
            band_list = self.bands

        ZP = 27.5
        # shift in time for visualization purposes
        t = self.data[self.bands[1]]['time'].min()
        tmax = str(t.astype(int))
        zeros = '0'*len(tmax[2:])
        t_off = int(tmax[:2] + zeros)

        # to set plot limits
        if plot_type=='flux':
            plot_lim_vals = np.array([change_zp(self.data[band]['flux'], self.data[band]['zp'], ZP)
                                                                                    for band in self.bands] + [0.0]
                                                                                    , dtype="object")
            ymin_lim = np.hstack(plot_lim_vals).min()*0.9
            if ymin_lim < 0.0:
                ymin_lim *= 1.1/0.9  # there might be some "negative" fluxes sometimes
            ymax_lim = np.hstack(plot_lim_vals).max()*1.05
        elif plot_type=='mag':
            plot_lim_vals = [[np.nanmin(self.data[band]['mag']), np.nanmax(self.data[band]['mag'])]
                                                                                            for band in self.bands]
            plot_lim_vals = np.ndarray.flatten(np.array(plot_lim_vals))
            ymin_lim = np.nanmin(plot_lim_vals)*0.98
            ymax_lim = np.nanmax(plot_lim_vals)*1.02

        fig, ax = plt.subplots(figsize=(8,6))
        for i, band in enumerate(band_list):
            if plot_type=='flux':
                y_norm = change_zp(1.0, self.data[band]['zp'], ZP)
                time = np.copy(self.data[band]['time'])
                flux, err = np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])
                flux, err = flux*y_norm, err*y_norm
                ax.errorbar(time-t_off, flux, err, fmt='o', mec='k', capsize=3, capthick=2, ms=8, elinewidth=3,
                                                                                    label=band, color=new_palette[i])
                ylabel = f'Flux (ZP = {ZP})'
            elif plot_type=='mag':
                ylabel = 'Apparent Magnitude'
                mask = np.where(self.data[band]['flux'] > 0)
                time = self.data[band]['time'][mask]
                mag, err = self.data[band]['mag'][mask], self.data[band]['mag_err'][mask]

                ax.errorbar(time-t_off, mag, err, fmt='o', mec='k', capsize=3, capthick=2, ms=8, elinewidth=3,
                                                                                    label=band, color=new_palette[i])

        ax.set_ylabel(ylabel, fontsize=16, family='serif')
        ax.set_xlabel(f'Time - {t_off} [days]', fontsize=16, family='serif')
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
                fig_name = f'{self.name}_lcs.png'
            #fig.tight_layout()
            plt.savefig(fig_name)

        plt.show()


    def normalize_data(self):
        """Normalizes the fluxes and zero-points (ZPs).

        Fluxes are converted to physical units by calculating the ZPs according to the
        magnitude system, for example: **AB**, **BD17** or **Vega**.
        """

        for band in self.bands:
            mag_sys = self.data[band]['mag_sys']
            current_zp = self.data[band]['zp']

            new_zp = calc_zp(self.filters[band]['wave'], self.filters[band]['transmission'],
                                        self.filters[band]['response_type'], mag_sys, band)

            self.data[band]['flux'] = change_zp(self.data[band]['flux'], current_zp, new_zp)
            self.data[band]['flux_err'] = change_zp(self.data[band]['flux_err'], current_zp, new_zp)
            self.data[band]['zp'] = new_zp

    ############################################################################
    ############################ Light Curves Fits #############################
    ############################################################################

    def fit_lcs(self, kernel='matern52', kernel2='matern52', fit_mag=True, min_time_extrap=-3, max_time_extrap=5,
                                                                        min_wave_extrap=-200, max_wave_extrap=200):
        """Fits the data for each band using gaussian process

        The time of rest-frame B-band peak luminosity is estimated by finding where the derivative is equal to zero.

        Parameters
        ----------
        kernel : str, default ``matern52``
            Kernel to be used in the **time**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        kernel2 : str, default ``squaredexp``
            Kernel to be used in the **wavelengt**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        fit_mag : bool, default ``True``
            If ``True``, the data is fitted in magnitude space (this is recommended for 2D fits). Otherwise, the data is
            fitted in flux space.
        min_time_extrap : int or float, default ``-3``
            Number of days the light-curve fit is extrapolated in the time axis with respect to first epoch.
        max_time_extrap : int or float, default ``5``
            Number of days the light-curve fit is extrapolated in the time axis with respect to last epoch.
        min_wave_extrap : int or float, default ``-200``
            Number of angstroms the light-curve fit is extrapolated in the wavelengths axis with respect to reddest
            wavelength. This depends on the reddest filter.
        max_wave_extrap : int or float, default ``200``
            Number of angstroms the light-curve fit is extrapolated in the wavelengths axis with respect to bluest
            wavelength. This depends on the bluest filter.
        """
        ########################
        ####### GP Fit #########
        ########################

        self.calc_pivot()

        flux_array = np.hstack([self.data[band]['flux'] for band in self.bands])
        flux_err_array = np.hstack([self.data[band]['flux_err'] for band in self.bands])

        time_array = np.hstack([self.data[band]['time'] for band in self.bands])
        wave_array = np.hstack([[self.filters[band]['eff_wave']]*len(self.data[band]['time']) for band in self.bands])

        # edges to extrapolate in time and wavelength
        time_edges = np.array([time_array.min()+min_time_extrap, time_array.max()+max_time_extrap])
        bands_waves = np.hstack([self.filters[band]['wave'] for band in self.bands])
        bands_edges = np.array([bands_waves.min()+min_wave_extrap, bands_waves.max()+max_wave_extrap])

        if fit_mag:
            mask = flux_array > 0.0  # prevents nan values
            # ZPs are set to 0.0 to retrieve flux values after the GP fit
            mag_array, mag_err_array = flux2mag(flux_array[mask], 0.0, flux_err_array[mask])
            time_array, wave_array = time_array[mask], wave_array[mask]

            timeXwave, lc_mean, lc_std, gp_results = gp_2d_fit(time_array, wave_array, mag_array, mag_err_array,
                                                                kernel1=kernel, kernel2=kernel2,
                                                                x1_edges=time_edges, x2_edges=bands_edges)
            lc_mean, lc_std = mag2flux(lc_mean, 0.0, lc_std)

        else:
            timeXwave, lc_mean, lc_std, gp_results = gp_2d_fit(time_array, wave_array, flux_array, flux_err_array,
                                            kernel1=kernel, kernel2=kernel2, x2_edges=bands_edges)

        self.lc_fits['timeXwave'], self.lc_fits['lc_mean'] = timeXwave, lc_mean
        self.lc_fits['lc_std'], self.lc_fits['gp_results'] = lc_std, gp_results

        ###############################
        ##### Estimate B-band Peak ####
        ###############################
        times, waves = timeXwave.T[0], timeXwave.T[1]

        wave_ind = np.argmin(np.abs(self.filters['Bessell_B']['eff_wave']*(1+self.z) - waves))
        eff_wave = waves[wave_ind]  # closest wavelength from the gp grid to the effective_wavelength*(1+z) of Bessell_B
        mask = waves==eff_wave

        time, flux, flux_err = times[mask], lc_mean[mask], lc_std[mask]

        try:
            peak_id = peak.indexes(flux, thres=.3, min_dist=len(time)//2)[0]
            self.tmax = self.tmax0 = np.round(time[peak_id], 2)

            phaseXwave = np.copy(timeXwave)
            phaseXwave.T[0] = (times - self.tmax)/(1+self.z)
            self.lc_fits['phaseXwave'] = phaseXwave
        except:
            raise ValueError(f'Unable to obtain an initial estimation of B-band peak for {self.name}\
                                                                                            (poor peak coverage)')

        ##################################
        ## Save individual light curves ##
        ##################################
        phases = phaseXwave.T[0]
        for band in self.bands:
            wave_ind = np.argmin(np.abs(self.filters[band]['eff_wave'] - waves))
            eff_wave = waves[wave_ind]  # closest wavelength from the gp grid to the effective wavelength of the band
            mask = waves==eff_wave

            time, phase, flux, flux_err = times[mask], phases[mask], lc_mean[mask], lc_std[mask]
            mag, mag_err = flux2mag(flux, self.data[band]['zp'], flux_err)
            self.lc_fits[band] = {'time':time, 'phase':phase, 'flux':flux, 'flux_err':flux_err,
                                    'mag':mag, 'mag_err':mag_err}

            # calculate observed time and magnitude of peak for each band
            try:
                peak_id = peak.indexes(flux, thres=.3, min_dist=len(time)//2)[0]
                self.lc_fits[band]['tmax'] = np.round(time[peak_id], 2)
                self.lc_fits[band]['mmax'] = mag[peak_id]
            except:
                self.lc_fits[band]['tmax'] = self.lc_fits[band]['mmax'] = np.nan

    def plot_fits(self, plot_together=False, plot_type='flux', save=False, fig_name=None):
        """Plots the light-curves fits results.

        Plots the observed data for each band together with the gaussian process fits. The initial B-band
        peak estimation is plotted. The final B-band peak estimation after light-curves corrections is
        also potted if corrections have been applied.

        Parameters
        ----------
        plot_together : bool, default ``False``
            If ``False``, plots the bands separately. Otherwise, all bands are plotted together.
        plot_type : str, default ``flux``
            Type of value used for the data: either ``mag`` or ``flux``.
        save : bool, default ``False``
            If ``True``, saves the plot into a file.
        fig_name : str, default ``None``
            Name of the saved plot. If ``None`` is used the name of the file will be '{``self.name``}_lc_fits.png'.
            Only works if ``save`` is set to ``True``.

        """

        new_palette = [plt.get_cmap('Dark2')(i) for i in np.arange(8)] + [plt.get_cmap('Set1')(i) for i in np.arange(8)]
        ZP = 27.5  # zeropoint for normalising the flux for visualization purposes

        # shift in time for visualization purposes
        tmax = str(self.tmax.astype(int))
        zeros = '0'*len(tmax[2:])
        t_off = int(tmax[:2] + zeros)

        if plot_together:
            # to set plot limits
            if plot_type=='flux':
                plot_lim_vals = np.array([change_zp(self.data[band]['flux'], self.data[band]['zp'], ZP)
                                                                                        for band in self.bands] + [0.0],
                                                                                        dtype="object")
                ymin_lim = np.hstack(plot_lim_vals).min()*0.9
                if ymin_lim < 0.0:
                    ymin_lim *= 1.1/0.9  # there might be some "negative" fluxes sometimes
                ymax_lim = np.hstack(plot_lim_vals).max()*1.05
            elif plot_type=='mag':
                plot_lim_vals = [[np.nanmin(self.data[band]['mag']), np.nanmax(self.data[band]['mag'])]
                                                                                                for band in self.bands]
                plot_lim_vals = np.ndarray.flatten(np.array(plot_lim_vals))
                ymin_lim = np.nanmin(plot_lim_vals)*0.98
                ymax_lim = np.nanmax(plot_lim_vals)*1.02

            fig, ax = plt.subplots(figsize=(8, 6))
            for i, band in enumerate(self.bands):

                # GP fits
                time = np.copy(self.lc_fits[band]['time'])
                flux, flux_err = np.copy(self.lc_fits[band]['flux']), np.copy(self.lc_fits[band]['flux_err'])
                mag, mag_err = np.copy(self.lc_fits[band]['mag']), np.copy(self.lc_fits[band]['mag_err'])
                # Data
                data_time = np.copy(self.data[band]['time'])
                data_flux, data_flux_err = np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])
                data_mag, data_mag_err = np.copy(self.data[band]['mag']), np.copy(self.data[band]['mag_err'])

                if plot_type=='flux':
                    y_norm = change_zp(1.0, self.data[band]['zp'], ZP)
                    flux, err = flux*y_norm, flux_err*y_norm
                    data_flux, data_flux_err = data_flux*y_norm, data_flux_err*y_norm

                    ax.errorbar(data_time-t_off, data_flux, data_flux_err, fmt='o', mec='k', capsize=3, capthick=2,
                                                    ms=8, elinewidth=3, color=new_palette[i],label=band)
                    ax.plot(time-t_off, flux,'-', color=new_palette[i], lw=2, zorder=16)
                    ax.fill_between(time-t_off, flux-flux_err, flux+flux_err, alpha=0.5, color=new_palette[i])
                    ax.set_ylabel(f'Flux (ZP = {ZP})', fontsize=16, family='serif')

                elif plot_type=='mag':
                    ax.errorbar(data_time-t_off, data_mag, data_mag_err, fmt='o', mec='k', capsize=3, capthick=2, ms=8,
                                elinewidth=3, color=new_palette[i],label=band)
                    ax.plot(time-t_off, mag,'-', color=new_palette[i], lw=2, zorder=16)
                    ax.fill_between(time-t_off, mag-mag_err, mag+mag_err, alpha=0.5, color=new_palette[i])
                    ax.set_ylabel(r'Apparent Magnitude', fontsize=16, family='serif')

            ax.axvline(x=self.tmax0-t_off, color='k', linestyle='--', alpha=0.4)
            ax.axvline(x=self.tmax-t_off, color='k', linestyle='--')
            ax.minorticks_on()
            ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True, labelsize=16)
            ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True, labelsize=16)
            ax.set_xlabel(f'Time - {t_off} [days]', fontsize=16, family='serif')

            ax.set_title(f'{self.name}\nz = {self.z:.5}', fontsize=18, family='serif')
            ax.legend(fontsize=13, loc='upper right')
            ax.set_ylim(ymin_lim, ymax_lim)

            if plot_type=='mag':
                plt.gca().invert_yaxis()
        # plot each band separately
        else:
            h = 3
            v = math.ceil(len(self.bands) / h)

            fig = plt.figure(figsize=(15, 5*v))
            gs = gridspec.GridSpec(v , h)

            for i, band in enumerate(self.bands):
                j = math.ceil(i % h)
                k =i // h
                ax = plt.subplot(gs[k,j])

                time = np.copy(self.lc_fits[band]['time'])
                flux, flux_err = np.copy(self.lc_fits[band]['flux']), np.copy(self.lc_fits[band]['flux_err'])
                mag, mag_err = np.copy(self.lc_fits[band]['mag']), np.copy(self.lc_fits[band]['mag_err'])
                # Data
                data_time = np.copy(self.data[band]['time'])
                data_flux, data_flux_err = np.copy(self.data[band]['flux']), np.copy(self.data[band]['flux_err'])
                data_mag, data_mag_err = np.copy(self.data[band]['mag']), np.copy(self.data[band]['mag_err'])

                if plot_type=='flux':
                    y_norm = change_zp(1.0, self.data[band]['zp'], ZP)
                    flux, flux_err = flux*y_norm, flux_err*y_norm
                    data_flux, data_flux_err = data_flux*y_norm, data_flux_err*y_norm

                    ax.errorbar(data_time-t_off, data_flux, data_flux_err, fmt='o', color=new_palette[i],
                                    capsize=3, capthick=2, ms=8, elinewidth=3, mec='k')
                    ax.plot(time-t_off, flux,'-', lw=2, zorder=16, color=new_palette[i])
                    ax.fill_between(time-t_off, flux-flux_err, flux+flux_err, alpha=0.5, color=new_palette[i])

                elif plot_type=='mag':
                    ax.errorbar(data_time-t_off, data_mag, data_mag_err, fmt='o', color=new_palette[i],
                                    capsize=3, capthick=2, ms=8, elinewidth=3, mec='k')
                    ax.plot(time-t_off, mag,'-', lw=2, zorder=16, color=new_palette[i])
                    ax.fill_between(time-t_off, mag-mag_err, mag+mag_err, alpha=0.5, color=new_palette[i])
                    ax.invert_yaxis()

                ax.axvline(x=self.tmax0-t_off, color='k', linestyle='--', alpha=0.4)
                ax.axvline(x=self.tmax-t_off, color='k', linestyle='--')
                ax.set_title(f'{band}', fontsize=16, family='serif')
                ax.xaxis.set_tick_params(labelsize=15)
                ax.yaxis.set_tick_params(labelsize=15)
                ax.minorticks_on()
                ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
                ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)

            fig.text(0.5, 0.95, f'{self.name} (z = {self.z:.5})', ha='center', fontsize=20, family='serif')
            fig.text(0.5, 0.04, f'Time - {t_off} [days]', ha='center', fontsize=18, family='serif')
            if plot_type=='flux':
                fig.text(0.04, 0.5, f'Flux (ZP = {ZP})', va='center', rotation='vertical', fontsize=18, family='serif')
            elif plot_type=='mag':
                fig.text(0.04, 0.5, r'Apparent Magnitude', va='center',
                 rotation='vertical', fontsize=18, family='serif')

        if save:
            if fig_name is None:
                fig_name = f'{self.name}_lc_fits.png'
            #fig.tight_layout()
            plt.savefig(fig_name)

        plt.show()

    ############################################################################
    ######################### Light Curves Correction ##########################
    ############################################################################

    def mangle_sed(self, min_phase=-15, max_phase=30, method='gp', kernel='squaredexp', linear_extrap=True,
                    correct_extinction=True, scaling=0.86, reddening_law='fitzpatrick99', dustmaps_dir=None,
                    r_v=3.1, ebv=None):
        """Mangles the SED with the given method to match the SN magnitudes.

        Parameters
        ----------
        min_phase : int, default ``-15``
            Minimum phase to mangle.
        max_phase : int, default ``30``
            Maximum phase to mangle.
        method : str, defult ``gp``
            Method to estimate the mangling function. Either ``gp`` or ``spline``.
        kernel : str, default ``squaredexp``
            Kernel to be used for the gaussian process fit of the mangling function.  E.g, ``matern52``,
            ``matern32``, ``squaredexp``.
        linear_extrap: bool, default ``True``
            Type of extrapolation for the edges. Linear if ``True``, free (gaussian process extrapolation) if ``False``.
        correct_extinction: bool, default ``True``
            Whether or not to correct for Milky Way extinction.
        scaling : float, default ``0.86``
            Calibration of the Milky Way dust maps. Either ``0.86``
            for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
            dust map of Schlegel, Fikbeiner & Davis (1998).
        reddening_law: str, default ``fitzpatrick99``
            Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989), ``odonnell94`` (O’Donnell 1994),
            ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00`` (Calzetti 2000) and ``fm07``
            (Fitzpatrick & Massa 2007 with :math:`R_V` = 3.1.)
        dustmaps_dir : str, default ``None``
            Directory where the dust maps of Schlegel, Fikbeiner & Davis (1998) are found.
        r_v : float, default ``3.1``
            Total-to-selective extinction ratio (:math:`R_V`)
        ebv : float, default ``None``
            Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.
        """

        phases = np.arange(min_phase, max_phase+1, 1)

        # save user inputs for later (used when checking B-band peak estimation)
        self.user_input['mangle_sed'] = {'min_phase':min_phase, 'max_phase':max_phase, 'method':method, 'kernel':kernel,
                                         'linear_extrap':linear_extrap, 'correct_extinction':correct_extinction,
                                         'scaling':scaling, 'reddening_law':reddening_law, 'dustmaps_dir':dustmaps_dir,
                                         'r_v':r_v, 'ebv':ebv}

        lc_phases = self.lc_fits[self.pivot_band]['phase']

        ####################################
        ##### Calculate SED photometry #####
        ####################################
        sed_df = self.sed['data'].copy()
        # to match the available epochs from the lcs
        sed_df = sed_df[(lc_phases.min() <= sed_df.phase) & (sed_df.phase <= lc_phases.max())]
        sed_df = sed_df[sed_df.phase.isin(phases)]  # to match the requested epochs

        # first redshift the SED ("move" it in z) and then apply extinction from MW only
        sed_df.wave, sed_df.flux = sed_df.wave.values*(1+self.z), sed_df.flux.values/(1+self.z)
        if correct_extinction:
            sed_df.flux = redden(sed_df.wave.values, sed_df.flux.values, self.ra, self.dec,
                                                            scaling, reddening_law, dustmaps_dir, r_v, ebv)
            if ebv is None:
                self.mw_ebv = calculate_ebv(self.ra, self.dec, scaling, dustmaps_dir)  # calculates MW reddening
            else:
                self.mw_ebv = ebv

        bands2mangle = []
        # check which bands are in the wavelength range of the SED template
        for band in self.bands:
            filter_wave = self.filters[band]['wave']
            if (filter_wave.min() > sed_df.wave.values.min()) & (filter_wave.max() < sed_df.wave.values.max()):
                bands2mangle.append(band)

        self.sed_lcs = {band:{'flux':[], 'time':None, 'phase':None} for band in bands2mangle}
        sed_phases = sed_df.phase.unique()

        # calculate SED light curves
        for phase in sed_phases:
            phase_df = sed_df[sed_df.phase==phase]
            for band in bands2mangle:
                band_flux = integrate_filter(phase_df.wave.values, phase_df.flux.values, self.filters[band]['wave'],
                                           self.filters[band]['transmission'], self.filters[band]['response_type'])
                self.sed_lcs[band]['flux'].append(band_flux)

        for band in bands2mangle:
            self.sed_lcs[band]['flux'] = np.array(self.sed_lcs[band]['flux'])
            self.sed_lcs[band]['phase'] = sed_phases
            self.sed_lcs[band]['time'] = sed_phases*(1+self.z) + self.tmax

        ###################################
        ####### set-up for mangling #######
        ###################################
        # find the fluxes at the exact SED phases
        obs_flux_dict = {band:np.interp(sed_phases, self.lc_fits[band]['phase'], self.lc_fits[band]['flux'],
                                        left=0.0, right=0.0) for band in bands2mangle}
        obs_err_dict = {band:np.interp(sed_phases, self.lc_fits[band]['phase'], self.lc_fits[band]['flux_err'],
                                        left=0.0, right=0.0) for band in bands2mangle}
        flux_ratios_dict = {band:obs_flux_dict[band]/self.sed_lcs[band]['flux'] for band in bands2mangle}
        flux_ratios_err_dict = {band:obs_err_dict[band]/self.sed_lcs[band]['flux'] for band in bands2mangle}

        wave_array = np.array([self.filters[band]['eff_wave'] for band in bands2mangle])
        bands_waves = np.hstack([self.filters[band]['wave'] for band in bands2mangle])
        # includes the edges of the reddest and bluest bands
        x_edges = np.array([bands_waves.min(), bands_waves.max()])

        ################################
        ########## mangle SED ##########
        ################################
        self.mangled_sed = pd.DataFrame(columns=['phase', 'wave', 'flux', 'flux_err'])
        for i, phase in enumerate(sed_phases):
            obs_fluxes = np.array([obs_flux_dict[band][i] for band in bands2mangle])
            obs_errs = np.array([obs_err_dict[band][i] for band in bands2mangle])
            flux_ratios_array = np.array([flux_ratios_dict[band][i] for band in bands2mangle])
            flux_ratios_err_array = np.array([flux_ratios_err_dict[band][i] for band in bands2mangle])

            phase_df = sed_df[sed_df.phase==phase]
            sed_epoch_wave, sed_epoch_flux = phase_df.wave.values, phase_df.flux.values

            # mangling routine including optimisation
            mangling_results = mangle(wave_array, flux_ratios_array, flux_ratios_err_array, sed_epoch_wave,
                                        sed_epoch_flux, obs_fluxes, obs_errs, bands2mangle, self.filters, method,
                                        kernel, x_edges, linear_extrap)

            # precision of the mangling function
            mag_diffs = {band:-2.5*np.log10(mangling_results['flux_ratios'][i]) if mangling_results['flux_ratios'][i]>0
                                                else np.nan for i, band in enumerate(bands2mangle)}
            self.mangling_results.update({phase:mangling_results})
            self.mangling_results[phase].update({'mag_diff':mag_diffs})

            # save the SED phase info into a DataFrame
            mangled_sed = mangling_results['mangled_sed']
            mangled_wave = mangled_sed['wave']
            mangled_flux, mangled_flux_err = mangled_sed['flux'], mangled_sed['flux_err']
            phase_info = np.array([[phase]*len(mangled_wave), mangled_wave, mangled_flux, mangled_flux_err]).T
            phase_df = pd.DataFrame(data=phase_info, columns=['phase', 'wave', 'flux', 'flux_err'])
            self.mangled_sed = pd.concat([self.mangled_sed, phase_df])  # updated mangled SED for a single epoch

        # correct mangled SED for MW extinction first and then de-redshift it ("move" it back in z)
        self.corrected_sed = self.mangled_sed.copy()
        if correct_extinction:
            self.corrected_sed.flux = deredden(self.corrected_sed.wave.values, self.corrected_sed.flux.values,
                                                                                    self.ra, self.dec, scaling,
                                                                                    reddening_law, dustmaps_dir,
                                                                                    r_v, ebv)
        self.corrected_sed.wave = self.corrected_sed.wave.values/(1+self.z)
        self.corrected_sed.flux = self.corrected_sed.flux.values*(1+self.z)


    def plot_mangling_function(self, phase=0, mangling_function_only=False, verbose=True, save=False, fig_name=None):
        """Plot the mangling function for a given phase.

        Parameters
        ----------
        phase : int, default ``0``
            Phase to plot the mangling function. By default it plots the mangling function at B-band peak.
        mangling_function_only : bool, default ``False``
            If ``True``, only plots the mangling function, otherwise, plots the SEDs and filters as well
            (with scaled values).
        verbose : bool, default ``True``
            If ``True``, returns the difference between the magnitudes from the fits and the magnitudes from the
            mangled SED, for each of the bands.
        save : bool, default ``False``
            If true, saves the plot into a file.
        fig_name : str, default ``None``
            Name of the saved plot. If ``None`` is used the name of the file will be
            '{``self.name``}_mangling_phase{``phase``}.png'. Only works if ``save`` is set to ``True``.

        """

        assert (phase in self.mangling_results.keys()), f'A mangling function was not calculated for phase {phase}.'

        man = self.mangling_results[phase]
        eff_waves = np.copy(man['init_flux_ratios']['waves'])
        init_flux_ratios = np.copy(man['init_flux_ratios']['flux_ratios'])

        opt_flux_ratios = np.copy(man['opt_flux_ratios']['flux_ratios'])
        obs_fluxes = np.copy(man['obs_band_fluxes']['fluxes'])
        sed_fluxes = np.copy(man['sed_band_fluxes']['fluxes'])

        x = np.copy(man['mangling_function']['waves'])
        y, yerr = np.copy(man['mangling_function']['flux_ratios']), np.copy(man['mangling_function']['flux_ratios_err'])
        mang_sed_wave, mang_sed_flux = man['mangled_sed']['wave'], man['mangled_sed']['flux']
        init_sed_wave, init_sed_flux = man['init_sed']['wave'], man['init_sed']['flux']

        kernel = man['kernel']
        bands = list(man['mag_diff'].keys())

        if mangling_function_only:
            fig, ax = plt.subplots(figsize=(8,6))
            ax2 = ax.twiny()

            exp = np.round(np.log10(init_flux_ratios.max()), 0)
            y_norm = 10**exp
            init_flux_ratios = init_flux_ratios/y_norm
            y, yerr = y/y_norm, yerr/y_norm
            opt_flux_ratios = opt_flux_ratios/y_norm

            ax.scatter(eff_waves, init_flux_ratios, marker='o', label='Initial values')
            ax.plot(x, y)
            ax.fill_between(x, y-yerr, y+yerr, alpha=0.5, color='orange')
            ax.scatter(eff_waves, opt_flux_ratios, marker='*', color='red', label='Optimized values')

            ax.set_xlabel(r'Observer-frame Wavelength [$\AA$]', fontsize=16, family='serif')
            ax.set_ylabel(r'(Flux$_{\rm Obs}$ / Flux$_{\rm Temp}) \times$ 10$^{%.0f}$'%exp, fontsize=16, family='serif')
            ax.minorticks_on()
            ax.tick_params(which='both', length=8, width=1, direction='in', right=True, labelsize=16)
            ax.tick_params(which='minor', length=4)
            ax.set_ylim(y.min()*0.95, y.max()*1.03)

            ax2.set_xticks(ax.get_xticks())
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
            ax3.plot(eff_waves, sed_fluxes2, 'ok', ms=14, label='Initial SED values',
                                                alpha=0.8, fillstyle='none', markeredgewidth=2)  # initial sed fluxes

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

            ax2.set_xticks(ax.get_xticks())
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
                fig_name = f'{self.name}_mangling_phase{phase}.png'
            #fig.tight_layout()
            plt.savefig(fig_name)

        plt.show()

        if verbose:
            print(f'Mangling results - difference between mangled SED and "observed" magnitudes at phase {phase}:')
            for band, diff in man['mag_diff'].items():
                print(f'{band}: {np.round(diff, 4):.4f} [mags]')


    def _calculate_corrected_lcs(self):
        """Calculates the SN light curves applying extinction and k-corrections.

        **Note:** this function is used inside :func:`self.calculate_lc_params()`
        """

        corrected_lcs = {}
        phases = self.corrected_sed.phase.unique()

        for band in self.filters.keys():
            band_flux, band_flux_err, band_phase = [], [], []
            for phase in phases:
                phase_df = self.corrected_sed[self.corrected_sed.phase==phase]

                phase_wave = phase_df.wave.values
                phase_flux = phase_df.flux.values
                phase_flux_err = phase_df.flux_err.values

                filter_data = self.filters[band]
                try:
                    band_flux.append(integrate_filter(phase_wave, phase_flux, filter_data['wave'],
                                                        filter_data['transmission'], filter_data['response_type']))
                    band_flux_err.append(integrate_filter(phase_wave, phase_flux_err, filter_data['wave'],
                                                            filter_data['transmission'], filter_data['response_type']))
                    band_phase.append(phase)
                except:
                    pass

            if 'Bessell_' in band:
                zp = calc_zp(self.filters[band]['wave'], self.filters[band]['transmission'],
                                            self.filters[band]['response_type'], 'BD17', band)
            else:
                zp = self.data[band]['zp']

            if len(band_flux)>0:
                band_flux, band_flux_err, band_phase = np.array(band_flux), np.array(band_flux_err), np.array(band_phase)
                band_mag, band_mag_err = flux2mag(band_flux, zp, band_flux_err)
                corrected_lcs[band] = {'phase':band_phase, 'flux':band_flux, 'flux_err':band_flux_err,
                                       'mag':band_mag, 'mag_err':band_mag_err, 'zp':zp}

        self.corrected_lcs = corrected_lcs

        # simple, independent 1D fit to the corrected light curves
        corrected_lcs_fit = {}
        for band in corrected_lcs.keys():
            phase, zp = corrected_lcs[band]['phase'], corrected_lcs[band]['zp']
            flux, flux_err = corrected_lcs[band]['flux'], corrected_lcs[band]['flux_err']

            phase_fit, flux_fit, _ = gp_lc_fit(phase, flux, flux*1e-3)
            flux_err_fit = np.interp(phase_fit, phase, flux_err, left=0.0, right=0.0)  # linear extrapolation of errors

            mag_fit, mag_err_fit = flux2mag(flux_fit, zp, flux_err_fit)
            corrected_lcs_fit[band] = {'phase':phase_fit, 'flux':flux_fit, 'flux_err':flux_err_fit,
                                       'mag':mag_fit, 'mag_err':mag_err_fit, 'zp':zp}

        self.corrected_lcs_fit = corrected_lcs_fit


    def calculate_lc_params(self, maxiter=5):
        """Calculates the light-curves parameters.

        Estimation of B-band peak apparent magnitude (m :math:`_B^{max}`), stretch (:math:`\Delta` m :math:`_{15}(B)`)
        and colour (:math:`(B-V)^{max}`) parameters. An interpolation of the corrected light curves is done as well as
        part of this process.

        Parameters
        ----------
        maxiter : int, default ``5``
            Maximum number of iteration of the correction process to estimate an accurate B-band peak.

        """

        self._calculate_corrected_lcs()

        ########################################
        ########### Check B-band max ###########
        ########################################
        bmax_needs_check = True
        iter = 0

        assert 'Bessell_B' in self.corrected_lcs_fit.keys(), 'The rest-frame B-band light curve was not calculated after\
                                                                        corrections. Not enough wavelength coverage.'

        while bmax_needs_check:
            # estimate offset between inital B-band peak and "final" peak
            try:
                b_data = self.corrected_lcs['Bessell_B']
                b_phase, b_flux, b_err = b_data['phase'], b_data['flux'], b_data['flux_err']
                b_phase, b_flux, b_err = gp_lc_fit(b_phase, b_flux, b_err)  # smoother estimation of the peak

                peak_id = peak.indexes(b_flux, thres=.3, min_dist=len(b_phase)//2)[0]
                phase_offset = b_phase[peak_id] - 0.0

                self._phase_offset = np.round(phase_offset, 2)
            except:
                phase_offset = None

            assert phase_offset is not None, "The time of rest-frame B-band peak luminosity can not be calculated. \
                                                Not enough time coverage."

            # error propagation
            try:
                b_data = self.corrected_lcs_fit['Bessell_B']
                b_phase, b_flux, b_err = b_data['phase'], b_data['flux'], b_data['flux_err']
                simulated_lcs = np.asarray([np.random.normal(flux, err, 1000) for flux, err in zip(b_flux, b_err)])

                pmax_list = []
                # loop to estimate uncertainty in tmax
                for lc_flux in simulated_lcs.T:
                    # the LC needs to be smoothed as the "simulations" are "noisy"
                    lc_flux = savgol_filter(lc_flux, 91, 3)
                    idx_max = peak.indexes(lc_flux, thres=.3, min_dist=len(b_phase)//2)[0]
                    pmax_list.append(b_phase[idx_max])

                pmax_array = np.array(pmax_list)
                self.tmax_err = pmax_array.std().round(2)
            except:
                self.tmax_err = np.nan

            if iter>=maxiter:
                break
            iter += 1

            # compare tmax from the corrected restframe B-band to the initial estimation
            if np.abs(phase_offset) >= 0.2:
                if np.abs(self.tmax0-self.tmax)/(1+self.z) >= 0.5:
                    self.tmax = np.copy(self.tmax0)  # back to initial estimation - going too far
                    phase_offset = random.uniform(-0.2, 0.2)

                # update phase of the light curves
                self.tmax = np.round(self.tmax - phase_offset*(1+self.z), 2)
                self.lc_fits['phaseXwave'].T[0] -= phase_offset
                for band in self.bands:
                    self.lc_fits[band]['phase'] -= phase_offset

                # re-do mangling
                self.mangle_sed(**self.user_input['mangle_sed'])
                self._calculate_corrected_lcs()
            else:
                bmax_needs_check = False

        ########################################
        ### Calculate Light Curve Parameters ###
        ########################################
        bessell_b = 'Bessell_B'

        # B-band peak apparent magnitude
        b_phase = self.corrected_lcs[bessell_b]['phase']
        b_mag, b_mag_err = self.corrected_lcs[bessell_b]['mag'], self.corrected_lcs[bessell_b]['mag_err']
        id_bmax = list(b_phase).index(0)

        mb, mb_err = b_mag[id_bmax], b_mag_err[id_bmax]

        # Stretch parameter
        if 15 in b_phase:
            id_15 = list(b_phase).index(15)
            B15, B15_err = b_mag[id_15], b_mag_err[id_15]

            dm15 = B15 - mb

            # covariance from the 2D gp fit to the light curves
            # gp_results = self.lc_fits['gp_results']
            # gp = gp_results['gp']
            # x1_norm, x2_norm, y_norm = gp_results['x1_norm'], gp_results['x2_norm'], gp_results['y_norm']
            #
            # x1_range = np.array([self.tmax, self.tmax + 15*(1+self.z)])/x1_norm
            #
            # eff_wave_B = self.filters[bessell_b]['eff_wave']*(1+self.z)
            # x2_range = np.array([eff_wave_B])/x2_norm
            #
            # X_predict = np.array(np.meshgrid(x1_range, x2_range)).reshape(2, -1).T
            # _, cov_matrix = gp(X_predict)
            # cov_B0_B15 = cov_matrix[0][1]*y_norm**2
            cov_B0_B15 = 0.0  # no covariance included

            dm15_err = np.sqrt(np.abs(mb_err**2 + B15_err**2 - 2*cov_B0_B15))

        else:
            dm15 = dm15_err = np.nan

        # Colour
        colour = colour_err = np.nan
        if 'Bessell_V' in self.corrected_lcs.keys():
            bessell_v = 'Bessell_V'
            if 0 in self.corrected_lcs[bessell_v]['phase']:
                v_phase = self.corrected_lcs[bessell_v]['phase']
                v_mag, v_mag_err = self.corrected_lcs[bessell_v]['mag'], self.corrected_lcs[bessell_v]['mag_err']

                id_v0 = list(v_phase).index(0)
                V0, V0_err = v_mag[id_v0], v_mag_err[id_v0]

                colour = mb - V0

                # covariance from the 2D gp fit to the light curves
                # gp_results = self.lc_fits['gp_results']
                # gp = gp_results['gp']
                # x1_norm, x2_norm, y_norm = gp_results['x1_norm'], gp_results['x2_norm'], gp_results['y_norm']
                #
                # x1_range = np.array([self.tmax])/x1_norm
                #
                # eff_wave_B = self.filters[bessell_b]['eff_wave']*(1+self.z)
                # eff_wave_V = self.filters[bessell_v]['eff_wave']*(1+self.z)
                # x2_range = np.array([eff_wave_B, eff_wave_V])/x2_norm
                #
                # X_predict = np.array(np.meshgrid(x1_range, x2_range)).reshape(2, -1).T
                # _, cov_matrix = gp(X_predict)
                # cov_B_V = cov_matrix[0][1]*y_norm**2
                cov_B_V = 0.0  # no covariance included

                colour_err = np.sqrt(np.abs(mb_err**2 + V0_err**2 - 2*cov_B_V))

        self.lc_parameters = {'mb':mb, 'mb_err':mb_err, 'dm15':dm15,
                              'dm15_err':dm15_err, 'colour':colour, 'colour_err':colour_err}


    def display_results(self, band='Bessell_B', plot_type='mag', display_params=False, save=False, fig_name=None):
        """Displays the rest-frame light curve for the given band.

        Plots the rest-frame band light curve together with a gaussian fit to it. The parameters estimated with
        :func:`calculate_lc_params()` are shown as well.

        Parameters
        ----------
        band : str, default ``Bessell_B``
            Name of the band to be plotted.
        plot_type : str, default ``mag``
            Type of value used for the data: either ``mag`` or ``flux``.
        display_params : bool, default ``False``
            If ``True``, the light-curves parameters are displayed in the plot.
        save : bool, default ``False``
            If ``True``, saves the plot into a file.
        fig_name : str, default ``None``
            Name of the saved plot. If ``None`` is used the name of the file will be
            '{``self.name``}_restframe_{``band``}.png'. Only works if ``save`` is set to ``True``.

        """

        assert (plot_type=='mag' or plot_type=='flux'), f'"{plot_type}" is not a valid plot type.'

        mb = self.lc_parameters['mb']
        mb_err = self.lc_parameters['mb_err']
        dm15 = self.lc_parameters['dm15']
        dm15_err = self.lc_parameters['dm15_err']
        colour = self.lc_parameters['colour']
        colour_err = self.lc_parameters['colour_err']

        if band is None:
            band = 'Bessell_B'

        x = np.copy(self.corrected_lcs[band]['phase'])
        y = np.copy(self.corrected_lcs[band][plot_type])
        yerr = np.copy(self.corrected_lcs[band][plot_type+'_err'])
        zp = self.corrected_lcs[band]['zp']

        x_fit = np.copy(self.corrected_lcs_fit[band]['phase'])
        y_fit = np.copy(self.corrected_lcs_fit[band][plot_type])
        yerr_fit = np.copy(self.corrected_lcs_fit[band][plot_type+'_err'])

        if plot_type=='flux':
            ZP = 27.5
            y_norm = change_zp(1.0, zp, ZP)
            y *= y_norm
            yerr *= y_norm
            y_fit *= y_norm
            yerr_fit *= y_norm

        fig, ax = plt.subplots(figsize=(8,6))
        ax.errorbar(x, y, yerr, fmt='-.o', color='k', ecolor='k', mec='k', capsize=3, capthick=2, ms=8,
                                                elinewidth=3, zorder=16)
        ax.plot(x_fit, y_fit, 'c-', alpha=0.7)
        ax.fill_between(x_fit, y_fit+yerr_fit, y_fit-yerr_fit, alpha=0.5, color='c')

        if display_params:
            ax.text(0.75, 0.9,r'm$_B^{\rm max}$=%.3f$\pm$%.3f'%(mb, mb_err), ha='center', va='center',
                                                fontsize=15, transform=ax.transAxes)
            if not np.isnan(dm15):
                ax.text(0.75, 0.8,r'$\Delta$m$_{15}$($B$)=%.3f$\pm$%.3f'%(dm15, dm15_err), ha='center', va='center',
                                                    fontsize=15, transform=ax.transAxes)
            if not np.isnan(colour):
                position = 0.7
                if np.isnan(dm15):
                    position = 0.8
                ax.text(0.75, position,r'($B-V$)$_{\rm max}$=%.3f$\pm$%.3f'%(colour, colour_err), ha='center',
                                                            va='center', fontsize=15, transform=ax.transAxes)

        ax.set_xlabel(f'Phase with respect to B-band peak [days]', fontsize=16, family='serif')
        tmax_str = r't$_{\rm max}$'
        ax.set_title(f'{self.name}\n{band}, z={self.z:.5}, {tmax_str}={self.tmax:.2f}', fontsize=16, family='serif')
        if plot_type=='flux':
            ax.set_ylabel(f'Flux (ZP = {ZP})', fontsize=16, family='serif')
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
                fig_name = f'{self.name}_restframe_{band}.png'
            #fig.tight_layout()
            plt.savefig(fig_name)

        plt.show()


    def do_magic(self):
        """Applies the whole correction process with default settings to obtain restframe light curves and
        light-curve parameters.

        **Note:** this is meant to be used for "quick" fits.
        """

        self.normalize_data()
        self.fit_lcs()
        self.mangle_sed()
        self.calculate_lc_params()
