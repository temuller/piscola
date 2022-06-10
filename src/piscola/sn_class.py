# -*- coding: utf-8 -*-
# This is the skeleton of PISCOLA, the main file

import piscola
from .lightcurves_class import lightcurves, generic_lightcurves
from .filters_class import multi_filters
from .sed_class import sed_template

from .filter_utils import integrate_filter, calc_eff_wave, calc_pivot_wave, calc_zp, filter_effective_range
from .gaussian_process import gp_lc_fit, gp_2d_fit
from .extinction_correction import redden, deredden, calculate_ebv
from .utils import flux2mag, mag2flux, change_zp

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

def call_sn(lc_file):
    """Loads a supernova from a file and initialises
    the :func:`sn` object. The object is initialised
    with all the necessary information like filters,
    fluxes, etc.

    Parameters
    ----------
    lc_file : str
        File with the SN info and light curves.

    Returns
    -------
    sn_obj : obj
        New :func:`sn` object.
    """
    err_message = f'File {lc_file} not found.'
    assert os.path.isfile(lc_file), err_message

    converters = {'name': str, 'z': float,
                  'ra': float, 'dec': float}
    name, z, ra, dec = pd.read_csv(lc_file, nrows=1,
                                   delim_whitespace=True,
                                   converters=converters).values[0]

    sn_obj = supernova(name, z, ra, dec, lc_file)

    return sn_obj

def load_sn(piscola_file):
    """Loads a :func:`sn` oject that was previously saved as a pickle file.

    Parameters
    ----------
    piscola_file : str
        File with the SN object saved with PISCOLA.

    Returns
    -------
    pickle.load(file) : obj
        :func:`sn` object previously saved as a pickle file.
    """
    err_message = f'File {piscola_file} not found.'
    assert os.path.isfile(piscola_file), err_message

    with open(os.path.join(path, name) + '.pisco', 'rb') as sn_file:
        sn_obj = pickle.load(sn_file)

    return sn_obj

class supernova(object):
    """Supernova class.
    """
    def __init__(self, name, z=0.0, ra=None, dec=None,
                 lc_file=None, template='conley09f'):
        self.name = name
        self.z = z
        self.ra = ra
        self.dec = dec
        self.init_fits = {}

        if not self.ra or not self.dec:
            print('Warning, RA and/or DEC not specified.')

        # add light curves and filters
        if lc_file:
            lcs_df = pd.read_csv(lc_file, delim_whitespace=True,
                            skiprows=2)
            self.lcs = lightcurves(lcs_df)
            self.filters = multi_filters(self.lcs.bands)

            # order bands by effective wavelength
            eff_waves = [self.filters[band]['eff_wave']
                         for band in self.filters.bands]
            sorted_idx = sorted(range(len(eff_waves)),
                                key=lambda k: eff_waves[k])
            sorted_bands = [self.filters.bands[x]
                            for x in sorted_idx]
            lc_bands = [band for band in sorted_bands
                          if band in lcs_df.band.values]
            self.bands = self.lcs.bands =lc_bands
            self.filters.bands = sorted_bands
            self._normalize_lcs()

        # add SED template
        self.sed = sed_template(z, ra, dec, template)
        self.sed.calculate_obs_lightcurves(self.filters)

    def __repr__(self):
        rep = (f'name: {self.name}, z: {self.z:.5}, '
               f'ra: {self.ra}, dec: {self.dec}\n')
        return rep

    def __getitem__(self, item):
        return getattr(self, item)

    def save_sn(self, path=None):
        """Saves a SN object into a pickle file

        Parameters
        ----------
        path: str, default ``None``
            Path where to save the SN file given the ``name``. If None,
            use current directory
        """
        if name is None:
            name = self.name

        if path is None:
            path = ''

        outfile = os.path.join(path, f'{self.name}.pisco')
        with open(outfile, 'wb') as pfile:
            pickle.dump(self, pfile, pickle.HIGHEST_PROTOCOL)

    def set_sed_template(self, template):
        """Sets the SED template to be used for the mangling function.

        Parameters
        ----------
        template : str
            Template name. E.g., ``conley09f``, ``jla``, etc.
        """
        self.sed.set_sed_template(template)
        self.sed.calculate_obs_lightcurves(self.filters)

    def _normalize_lcs(self):
        """Normalizes the fluxes and zero-points (ZPs).

        Fluxes are converted to physical units by calculating the ZPs according to the
        magnitude system, for example: **AB**, **BD17** or **Vega**.
        """
        for band in self.bands:
            mag_sys = self.lcs[band].mag_sys
            current_zp = self.lcs[band].zp

            new_zp = self.filters[band].calc_zp(mag_sys)

            self.lcs[band].flux = change_zp(self.lcs[band]['flux'],
                                               current_zp, new_zp)
            self.lcs[band].flux_err = change_zp(self.lcs[band]['flux_err'],
                                                   current_zp, new_zp)
            self.lcs[band].zp = new_zp

    def _stack_lcs(self):
        """For 2D fitting.
        """
        for band in self.lcs.bands:
            # mask negative fluxes
            mask = ~np.isnan(self.lcs[band].mag)
            self.lcs[band].mask_lc(mask)

        time = np.hstack([self.lcs[band].masked_time for band in self.bands])
        wave = np.hstack([[self.filters[band].eff_wave]*len(self.lcs[band].masked_time)
                          for band in self.bands])
        flux = np.hstack([self.lcs[band].masked_flux for band in self.bands])
        flux_err = np.hstack([self.lcs[band].masked_flux_err for band in self.bands])
        mag = np.hstack([self.lcs[band].masked_mag for band in self.bands])
        mag_err = np.hstack([self.lcs[band].masked_mag_err for band in self.bands])

        self._stacked_time = time
        self._stacked_wave = wave
        self._stacked_flux = flux
        self._stacked_flux_err = flux_err
        self._stacked_mag = mag
        self._stacked_mag_err = mag_err

    def _fit_lcs(self, kernel1='matern52', kernel2='squaredexp', gp_mean='max'):
        """Fits the data for each band using gaussian process

        The time of rest-frame B-band peak luminosity is estimated by finding where the derivative is equal to zero.

        Parameters
        ----------
        kernel : str, default ``matern52``
            Kernel to be used in the **time**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        kernel2 : str, default ``matern52``
            Kernel to be used in the **wavelengt**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        """
        self._stack_lcs()
        timeXwave, lc_mean, lc_std, gp_pred, gp = gp_2d_fit(self._stacked_time,
                                                            self._stacked_wave,
                                                            self._stacked_mag,
                                                            self._stacked_mag_err,
                                                            kernel1, kernel2, gp_mean)

        self.init_fits['timeXwave'], self.init_fits['lc_mean'] = timeXwave, lc_mean
        self.init_fits['lc_std'], self.init_fits['gp_pred'] = lc_std, gp_pred
        self.init_fits['gp']= gp

        # Estimate B-band Peak
        sed_wave, sed_flux = self.sed.get_phase_data(0.0)
        B_eff_wave = self.filters.Bessell_B.calc_eff_wave(sed_wave, sed_flux)

        times, waves = timeXwave.T[0], timeXwave.T[1]
        wave_ind = np.argmin(np.abs(B_eff_wave*(1+self.z) - waves))
        eff_wave = waves[wave_ind]
        Bmask = waves == eff_wave
        Btime, Bmag = times[Bmask], lc_mean[Bmask]

        peak_id = peak.indexes(-Bmag, thres=.3, min_dist=len(Btime)//2)[0]
        self.init_tmax = np.round(Btime[peak_id], 2)

    def fit(self, kernel1='matern52', kernel2='squaredexp', gp_mean='mean'):
        self._fit_lcs(kernel1, kernel2, gp_mean)  # to get initial tmax

        sed_lcs = self.sed.obs_lcs_fit  # interpolated light curves
        sed_times = sed_lcs.phase.values + self.init_tmax
        flux_ratios = []
        flux_err = []
        for band in self.bands:
            sed_flux = np.interp(self.lcs[band].masked_time, sed_times,
                                 sed_lcs[band].values, left=0.0, right=0.0)
            flux_ratios.append(self.lcs[band].masked_flux/sed_flux)
            flux_err.append(self.lcs[band].masked_flux_err/sed_flux)
        stacked_ratios = np.hstack(flux_ratios)
        stacked_err = np.hstack(flux_err)
        timeXwave, mf_mean, mf_std, gp_pred, gp = gp_2d_fit(self._stacked_time,
                                                            self._stacked_wave,
                                                            stacked_ratios,
                                                            stacked_err,
                                                            kernel1, kernel2,
                                                            gp_mean)
        self.fit_results = {'timeXwave':timeXwave, 'mf_mean':mf_mean,
                            'mf_std':mf_std, 'gp_pred':gp_pred, 'gp':gp}

        times, waves = timeXwave.T
        fits_df_list = []
        for band in self.bands:
            wave_ind = np.argmin(np.abs(self.filters[band]['eff_wave'] - waves))
            eff_wave = waves[wave_ind]
            mask = waves == eff_wave
            time, mf, std = times[mask], mf_mean[mask], mf_std[mask]

            sed_flux = np.interp(time, sed_times, sed_lcs[band].values, 
                                 left=0.0, right=0.0)
            lc_fit = sed_flux*mf
            lc_std = sed_flux*std

            fit_df = pd.DataFrame({'time':time, 'flux':lc_fit,
                                   'flux_err':lc_std})
            fit_df['zp'] = self.lcs[band].zp
            fit_df['band'] = band
            fit_df['mag_sys'] = self.lcs[band].mag_sys
            fits_df_list.append(fit_df)

        self.lc_fits = lightcurves(pd.concat(fits_df_list))
        self._mangle_sed()
        self._get_rest_lightcurves()

    def _mangle_sed(self):
        times, waves = self.fit_results['timeXwave'].T
        mf_mean = self.fit_results['mf_mean']
        mf_std = self.fit_results['mf_std']

        # mangle SED in observer frame
        mangled_sed = {'flux': [], 'flux_err': []}
        for phase in np.unique(self.sed.phase):
            # SED
            phase_mask = self.sed.phase == phase
            sed_wave = self.sed.wave[phase_mask]
            sed_flux = self.sed.flux[phase_mask]

            # mangling function
            phases = (times - self.init_tmax)
            phase_id = np.argmin(np.abs(phases - phase))
            phase_mask = phases == phases[phase_id]
            mang_wave = waves[phase_mask]
            mang_func = mf_mean[phase_mask]
            mang_func_std = mf_std[phase_mask]

            mang_func = np.interp(sed_wave, mang_wave, mang_func)
            mang_func_std = np.interp(sed_wave, mang_wave, mang_func_std)
            mangled_flux = mang_func * sed_flux
            mangled_flux_err = mang_func_std * sed_flux

            mangled_sed['flux'] += list(mangled_flux)
            mangled_sed['flux_err'] += list(mangled_flux_err)

        self.sed.flux = np.array(mangled_sed['flux'])
        self.sed.flux_err = np.array(mangled_sed['flux_err'])

    def _get_rest_lightcurves(self):

        self.sed.calculate_rest_lightcurves(self.filters)
        lcs_df_list = []
        fits_df_list = []
        for band in self.filters.bands:
            if 'Bessell' in band:
                mag_sys = 'VEGA'
            else:
                mag_sys = self.lcs[band].mag_sys
            zp = self.filters[band].calc_zp(mag_sys)

            lc = self.sed.rest_lcs
            lc_df = pd.DataFrame({'time': lc.phase.values,
                                  'flux': lc[band].values,
                                  'flux_err': lc[f'{band}_err'].values
                                  })
            lc_df['zp'] = zp
            lc_df['band'] = band
            lc_df['mag_sys'] = mag_sys
            lcs_df_list.append(lc_df)

            fit = self.sed.rest_lcs_fit
            fit_df = pd.DataFrame({'time': fit.phase.values,
                                  'flux': fit[band].values,
                                  'flux_err': [0.0]*len(fit.phase.values)
                                  })
            fit_df['zp'] = zp
            fit_df['band'] = band
            fit_df['mag_sys'] = mag_sys
            fits_df_list.append(fit_df)

        self.rest_lcs = lightcurves(pd.concat(lcs_df_list))
        self.rest_lcs_fits = lightcurves(pd.concat(fits_df_list))

    def plot_fits(self, plot_mag=False, fig_name=None):
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
        palette1 = [plt.get_cmap('Dark2')(i) for i in np.arange(8)]
        palette2 = [plt.get_cmap('Set1')(i) for i in np.arange(8)]
        colours = palette1 + palette2

        # shift in time for visualization purposes
        tmax_str = str(self.init_tmax.astype(int))
        zeros = '0'*len(tmax_str[2:])
        t_offset = int(tmax_str[:2] + zeros)

        ZP = 27.5  # global zero-point for visualization

        h = 3  # columns
        v = math.ceil(len(self.bands) / h)  # rows

        fig = plt.figure(figsize=(15, 5 * v))
        gs = gridspec.GridSpec(v * 2, h, height_ratios=[3, 1] * v)

        for i, band in enumerate(self.bands):
            j = math.ceil(i % h)
            k = i // h * 2
            ax = plt.subplot(gs[k, j])
            ax2 = plt.subplot(gs[k + 1, j])

            x = self.lcs[band].time - t_offset
            x_fit = self.lc_fits[band].time - t_offset
            if not plot_mag:
                y_norm = change_zp(1.0, self.lcs[band]['zp'], ZP)
                y = self.lcs[band].flux * y_norm
                yerr = self.lcs[band].flux_err * y_norm
                y_fit = self.lc_fits[band].flux * y_norm
                yerr_fit = self.lc_fits[band].flux_err * y_norm
            else:
                y = self.lcs[band].mag
                yerr = self.lcs[band].mag_err
                y_fit = self.lc_fits[band].mag
                yerr_fit = self.lc_fits[band].mag_err
                ax.invert_yaxis()
                ax2.invert_yaxis()

            colour = colours[i]
            data_par = dict(fmt='o', color=colour, capsize=3,
                            capthick=2, ms=8, elinewidth=3, mec='k')
            fit_par = dict(ls='-', lw=2, zorder=16, color=colour)
            fit_err_par = dict(alpha=0.5, color=colour)

            # light curves
            ax.errorbar(x, y, yerr, label=band, **data_par)
            ax.plot(x_fit, y_fit, **fit_par)
            ax.fill_between(x_fit, y_fit - yerr_fit, y_fit + yerr_fit,
                            **fit_err_par)

            # residuals
            res = y - np.interp(x, x_fit, y_fit)
            ax2.errorbar(x, res, yerr, **data_par)
            ax2.plot(x_fit, np.zeros_like(y_fit), **fit_par)
            ax2.fill_between(x_fit, -yerr_fit, yerr_fit, **fit_err_par)

            for axis in [ax, ax2]:
                axis.axvline(x=self.init_tmax - t_offset, color='k',
                             linestyle='--', alpha=0.4)
                axis.xaxis.set_tick_params(labelsize=15)
                axis.yaxis.set_tick_params(labelsize=15)
                axis.minorticks_on()
                axis.tick_params(which='major', length=6, width=1,
                                 direction='in', top=True, right=True)
                axis.tick_params(which='minor', length=3, width=1,
                                 direction='in', top=True, right=True)
            ax.set_xticks([])
            ax.legend(fontsize=16)

        fig.text(0.5, 0.92, f'{self.name} (z = {self.z:.5})', ha='center',
                 fontsize=20, family='serif')
        fig.text(0.5, 0.05, f'Time - {t_offset} [days]', ha='center',
                 fontsize=18, family='serif')
        if not plot_mag:
            fig.text(0.05, 0.5, f'Flux (ZP = {ZP})', va='center',
                     rotation='vertical', fontsize=18, family='serif')
        else:
            fig.text(0.05, 0.5, r'Apparent Magnitude', va='center',
                     rotation='vertical', fontsize=18, family='serif')

        if fig_name is not None:
            plt.savefig(fig_name)

        plt.show()

################################################################################
################################################################################
################################################################################

# This is the main class
class sn(object):
    """Supernova class for representing a supernova."""

    def __init__(self, name, z=0, ra=None, dec=None):
        self.name = name
        self.z = z  # redshift
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


    ############################################################################
    ############################ Light Curves Fits #############################
    ############################################################################

    def fit_lcs(self, kernel='matern52', kernel2='squaredexp', gp_mean='max'):
        """Fits the data for each band using gaussian process

        The time of rest-frame B-band peak luminosity is estimated by finding where the derivative is equal to zero.

        Parameters
        ----------
        kernel : str, default ``matern52``
            Kernel to be used in the **time**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        kernel2 : str, default ``matern52``
            Kernel to be used in the **wavelengt**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        """
        ########################
        ####### GP Fit #########
        ########################
        stacked_flux = np.hstack([self.lcs[band].flux
                                  for band in self.bands])
        stacked_flux_err = np.hstack([self.lcs[band].flux_err
                                      for band in self.bands])

        stacked_time = np.hstack([self.lcs[band].time
                                  for band in self.bands])
        stacked_wave = np.hstack([[self.filters[band].eff_wave] * len(self.lcs[band].time)
                                  for band in self.bands])

        mask = stacked_flux > 0.0  # prevents nan values
        # ZPs are set to 0.0 to retrieve flux values after the GP fit
        stacked_mag, stacked_mag_err = flux2mag(stacked_flux[mask], 0.0,
                                                stacked_flux_err[mask])
        stacked_time, stacked_wave = stacked_time[mask], stacked_wave[mask]

        timeXwave, lc_mean, lc_std, gp_pred, gp = gp_2d_fit(stacked_time, stacked_wave,
                                                            stacked_mag, stacked_mag_err,
                                                            kernel1, kernel2, gp_mean)

        self.init_fits['timeXwave'], self.init_fits['lc_mean'] = timeXwave, lc_mean
        self.init_fits['lc_std'], self.init_fits['gp_pred'] = lc_std, gp_pred
        self.init_fits['gp']= gp

        # Estimate B-band Peak
        sed_wave, sed_flux = self.sed.get_phase_data(0.0)
        B_eff_wave = self.filters.Bessell_B.calc_eff_wave(sed_wave, sed_flux)

        times, waves = timeXwave.T[0], timeXwave.T[1]
        wave_ind = np.argmin(np.abs(B_eff_wave*(1+self.z) - waves))
        eff_wave = waves[wave_ind]
        mask = waves == eff_wave
        time, mag = times[mask], lc_mean[mask]

        peak_id = peak.indexes(-mag, thres=.3, min_dist=len(time)//2)[0]
        self.init_tmax = np.round(time[peak_id], 2)


    def nada(self):
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
        self.corrected_sed = self.mangled_sed.astype('float64').copy()
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

        **Note:** this function is used inside :func:`calculate_lc_params()`
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
            dm15_err = np.sqrt(np.abs(mb_err**2 + B15_err**2))

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
                colour_err = np.sqrt(np.abs(mb_err**2 + V0_err**2))

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


    def export_fits(self, output_file=None):
        """Exports the light-curve fits into an output file.

        Parameters
        ----------
        output_file : str, default ``None``
            Name of the output file.
        """

        if output_file is None:
            output_file = f'{self.name}_fits.dat'

        df_list = []
        columns = ['time', 'phase', 'flux', 'flux_err',
                           'mag', 'mag_err', 'zp', 'band']

        for band in self.bands:
            band_info = self.lc_fits[band]
            zp = self.data[band]['zp']
            band_info['zp'] = zp
            # dictionary for rounding numbers for pretty output
            rounding_dict = {key:3 if 'flux' not in key else 99 for
                                                 key in band_info.keys() }
            band_info['band'] = band

            # dataframe
            band_df = pd.DataFrame(band_info)
            band_df = band_df.round(rounding_dict)
            df_list.append(band_df[columns])

        # concatenate the dataframes for all the bands for exporting
        df_fits = pd.concat(df_list)
        df_fits.to_csv(output_file, sep='\t', index=False)


    def export_restframe_lcs(self, output_file=None):
        """Exports the corrected, rest-frame light-curves into an output file.

        Parameters
            ----------
            output_file : str, default ``None``
                Name of the output file.
        """

        if output_file is None:
                output_file = f'{self.name}_restframe_lcs.dat'

        df_list = []
        columns = ['phase', 'flux', 'flux_err',
                           'mag', 'mag_err', 'zp', 'band']

        for band in self.bands:
            band_info = self.corrected_lcs[band]
            # dictionary for rounding numbers for pretty output
            rounding_dict = {key:3 if 'flux' not in key else 99 for
                                                 key in band_info.keys() }
            band_info['band'] = band

            # dataframe
            band_df = pd.DataFrame(band_info)
            band_df = band_df.round(rounding_dict)
            df_list.append(band_df[columns])

        # concatenate the dataframes for all the bands for exporting
        df_fits = pd.concat(df_list)
        df_fits.to_csv(output_file, sep='\t', index=False)
