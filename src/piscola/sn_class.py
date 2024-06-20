# -*- coding: utf-8 -*-
# This is the skeleton of PISCOLA, the main file
import os
import bz2
import sys
import math
import numpy as np
import pandas as pd
from peakutils import peak
if sys.version_info.minor < 8:
    import pickle5 as pickle
else:
    # pickle is a built-in package starting from python v3.8
    import pickle

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .utils import change_zp, flux_to_mag, mag_to_flux
from .sed_class import SEDTemplate
from .filters_class import MultiFilters
from .lightcurves_class import Lightcurves
from .gaussian_process import prepare_gp_inputs, fit_gp_model

import jax
jax.config.update("jax_enable_x64", True)

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
    err_message = f"File {lc_file} not found."
    assert os.path.isfile(lc_file), err_message

    converters = {"name": str, "z": float, "ra": float, "dec": float}
    name, z, ra, dec = pd.read_csv(
        lc_file, nrows=1, sep='\s+', converters=converters
    ).values[0]

    sn_obj = Supernova(name, z, ra, dec, lc_file)

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
    err_message = f"File {piscola_file} not found."
    assert os.path.isfile(piscola_file), err_message

    with bz2.BZ2File(piscola_file, "rb") as sn_file:
        sn_obj = pickle.load(sn_file)

    sn_obj.sed._retrieve_template(sn_obj.sed.name)

    return sn_obj


class Supernova(object):
    """Class representing a supernova object."""

    def __init__(
        self, name, z=0.0, ra=None, dec=None, lc_file=None, template="csp"
    ):
        """
        Parameters
        ----------
        name: str
            Name of the supernova.
        z: float, default ``0.0``
            Redshift of the supernova.
        ra: float, default ``None``
            Right ascension of the supernova.
        dec: float, default ``None``
            Declination of the supernova.
        lc_file: str, default ``None``
            File with the supernova light-curve data.
        template: str, default ``csp``
            Name of the spectral energy distribution (SED) template.
            E.g., ``csp``, ``hsiao``, ``salt2``, ``salt3``, etc.
        """
        self.name = name
        self.z = z
        self.ra = ra
        self.dec = dec

        if not self.ra or not self.dec:
            print("Warning, RA and/or DEC not specified.")

        # add light curves and filters
        if lc_file:
            lcs_df = pd.read_csv(lc_file, sep='\s+', skiprows=2)
            # saves original light curves untouched
            self.init_lcs = Lightcurves(lcs_df)
            # these are the light curves that will be used
            self.lcs = Lightcurves(lcs_df)
            mag_systems = [self.lcs[band].mag_sys for band in self.lcs.bands]
            self.filters = MultiFilters(self.lcs.bands, mag_systems)

            # order bands by effective wavelength
            eff_waves = [self.filters[band]["eff_wave"] for band in self.filters.bands]
            sorted_idx = sorted(range(len(eff_waves)), key=lambda k: eff_waves[k])
            sorted_bands = [self.filters.bands[x] for x in sorted_idx]
            lc_bands = [band for band in sorted_bands if band in lcs_df.band.values]
            self.bands = self.lcs.bands = lc_bands
            self.filters.bands = sorted_bands
            self._normalize_lcs()

        # add SED template
        self.sed = SEDTemplate(z, ra, dec, template)

    def __repr__(self):
        rep = (
            f"name: {self.name}, z: {self.z:.5}, "
            f"ra: {self.ra:.5}, dec: {self.dec:.5}\n"
        )
        return rep

    def __getitem__(self, item):
        return getattr(self, item)
    
    def add_filter(self, band, mag_sys):
        """Adds a new filter.

        Parameters
        ----------
        band : str
            Name of the filter
        mag_sys : str
            Magnitude system.
        """
        self.filters._add_filter(band, mag_sys)
        # order bands by effective wavelength
        eff_waves = [self.filters[band]["eff_wave"] for band in self.filters.bands]
        sorted_idx = sorted(range(len(eff_waves)), key=lambda k: eff_waves[k])
        sorted_bands = [self.filters.bands[x] for x in sorted_idx]
        self.filters.bands = sorted_bands

    def save(self, path=None):
        """Saves a SN object into a pickle file

        Parameters
        ----------
        path: str, default ``None``
            Path where to save the SN object. If ``None``,
            use current directory. The file will be saved with
            the name ``<SN name>.pisco``.
        """
        self.sed.data = None  # to save space
        if path is None:
            path = ""

        outfile = os.path.join(path, f"{self.name}.pisco")
        with bz2.BZ2File(outfile, "wb") as pfile:
            protocol = 4
            pickle.dump(self, pfile, protocol)

        self.sed._retrieve_template(self.sed.name)

    def set_sed_template(self, template):
        """Sets the SED template to be used for the mangling function.

        Parameters
        ----------
        template : str
            Template name. E.g., ``csp``, ``hsiao``, ``salt2``, ``salt3``, etc.
        """
        self.sed._set_sed_template(template)

    def _normalize_lcs(self):
        """Normalizes the fluxes and zero-points (ZPs).

        Fluxes are converted to physical units by calculating the ZPs according to the
        magnitude system, for example: **AB**, **BD17** or **Vega**.
        """
        for band in self.bands:
            mag_sys = self.lcs[band].mag_sys
            current_zp = self.lcs[band].zp

            new_zp = self.filters[band].calculate_zp(mag_sys)

            self.lcs[band].fluxes = change_zp(self.lcs[band].fluxes, current_zp, new_zp)
            self.lcs[band].flux_errors = change_zp(
                self.lcs[band].flux_errors, current_zp, new_zp
            )
            self.lcs[band].zp = new_zp

    ####################
    # Light-curve fits #
    ####################
    def _stack_lcs(self, bands=None):
        """Stacks of light-curve properties

        Times, wavelengths, fluxes, magnitudes and errors are
        stacked for 2D fitting.

        Parameters
        ----------
        bands: list-like
            Bands used for the stacking.
        """
        if bands is None:
            stacking_bands = self.bands
        else:
            stacking_bands = bands

        # convert the data from multiple light curves into single arrays
        times = np.hstack([self.lcs[band].times for band in stacking_bands])
        wavelengths = np.hstack(
            [
                [self.filters[band].eff_wave] * len(self.lcs[band].times)
                for band in stacking_bands
            ]
        )
        fluxes = np.hstack([self.lcs[band].fluxes for band in stacking_bands])
        flux_errors = np.hstack([self.lcs[band].flux_errors for band in stacking_bands])
        magnitudes = np.hstack([self.lcs[band].magnitudes for band in self.bands])
        mag_errors = np.hstack([self.lcs[band].mag_errors for band in stacking_bands])

        # store in hidden variables for GP fitting
        self._stacked_times = times
        self._stacked_wavelengths = wavelengths
        self._stacked_fluxes = fluxes
        self._stacked_flux_errors = flux_errors
        self._stacked_magnitudes = magnitudes
        self._stacked_mag_errors = mag_errors

    def init_gp_predict(self, times_pred, wavelengths_pred, return_cov=True):
        """Returns the Gaussian Process prediction of the initial light-curve fits.

        Parameters
        ----------
        times_pred : ndarray
            Time-axis grid.
        wavelengths_pred : ndarray
            Wavelength-axis grid.
        return_cov : bool, optional
            Whether to return the covariance or variance.

        Returns
        -------
        mu: ndarray
            Mean of the Gaussian Process.
        var or cov: ndarray
            Either the variance or covariance of the Gaussian Process.
        """
        X_test, y, _, y_norm = prepare_gp_inputs(times_pred, wavelengths_pred, 
                                                self._stacked_fluxes, self._stacked_flux_errors, 
                                                fit_log=self.init_fit_log,
                                                wave_log=self.init_wave_log)
        # GP prediction
        if return_cov is True:
            mu, cov = self.init_gp_model.predict(y, X_test=X_test, return_cov=True)
            # renormalise outputs
            if self.init_fit_log is False:
                # flux space
                mu *= y_norm
                cov *= y_norm ** 2
            else:
                # convert logarithmic to flux space
                mu = 10 ** (mu - y_norm)
                cov *= np.abs(np.outer(mu, mu)) * np.log(10) ** 2
            return mu, cov
        else:
            mu, var = self.init_gp_model.predict(y, X_test=X_test, return_var=True)
            # renormalise outputs
            if self.init_fit_log is False:
                # flux space
                mu *= y_norm
                var *= y_norm ** 2
            else:
                # convert logarithmic to flux space
                mu = 10 ** (mu - y_norm)
                var *= np.abs(mu) ** 2 * np.log(10) ** 2
            return mu, var
        
    def fit_lcs(self, bands=None, k1='Matern52', fit_log=False, wave_log=False):
        """Fits the multi-colour observed light-curve data with Gaussian Process.

        The time of rest-frame B-band peak luminosity is estimated by finding where the derivative is equal to zero.

        Parameters
        ----------
        bands : list-like, default ``None``
            Bands used for fitting light curves. By default, use all the available bands.
        k1: str
            Kernel to be used for the time axis. Either ``Matern52``,
            ``Matern32`` or ``ExpSquared``.
        fit_log: bool, default ``False``.
            Whether to fit the light curves in logarithmic (base 10) scale.
        wave_log: bool, default ``False``.
            Whether to use logarithmic (base 10) scale for the 
            wavelength axis.
        """
        ##########
        # GP fit #
        ##########
        self._stack_lcs(bands)
        gp_model = fit_gp_model(self._stacked_times, self._stacked_wavelengths, 
                                self._stacked_fluxes, self._stacked_flux_errors, k1=k1, 
                                fit_log=fit_log, wave_log=wave_log)
        self.init_gp_model = gp_model  # store GP model
        self.init_k1 = k1
        self.init_fit_log = fit_log
        self.init_wave_log = wave_log

        ########################
        # Estimate B-band Peak #
        ########################
        obs_eff_wave = self.filters.Bessell_B.eff_wave * (1 + self.z)
        dt = 0.1 * (1 + self.z)  # 0.1 days in rest-frame, moved to observer frame
        times_pred = np.arange(self._stacked_times.min() - 5, 
                               self._stacked_times.max() + 5,
                               dt
                               )
        # store time array for the prediction in 'self.fit()'
        self.times_pred = times_pred
        wavelengths_pred = np.zeros_like(times_pred) + obs_eff_wave
        mu, cov = self.init_gp_predict(times_pred, wavelengths_pred, return_cov=True)

        # monte-carlo sampling
        lcs_sample = np.random.multivariate_normal(mu, cov, size=5000)
        tmax_list = []
        for lc in lcs_sample:
            peak_ids = peak.indexes(lc, thres=0.3, min_dist=len(times_pred) // 2)
            if len(peak_ids) == 0:
                # if no peak is found, just use the maximum
                max_id = np.argmax(lc)
            else:
                max_id = peak_ids[0]
            tmax_list.append(times_pred[max_id])
        # save time of maximum
        self.init_tmax = np.round(np.nanmean(tmax_list), 3)
        self.init_tmax_err = np.round(np.nanstd(tmax_list), 3)

        # get B-band parameters
        self._get_init_rest_lightcurves()

        ###########################
        # Inital light-curve fits #
        ###########################
        fitting_bands = bands
        if bands is None:
            fitting_bands = self.bands
            
        fits_df_list = []
        for band in fitting_bands:
            eff_wave = self.filters[band]["eff_wave"]
            wavelengths_pred = np.zeros_like(times_pred) + eff_wave
            mu, var = self.init_gp_predict(times_pred, wavelengths_pred, return_cov=False)
            std = np.sqrt(var)

            # store light-curve fits
            fit_df = pd.DataFrame({"time": times_pred, "flux": mu, "flux_err": std})
            fit_df["zp"] = self.lcs[band].zp
            fit_df["band"] = band
            fit_df["mag_sys"] = self.lcs[band].mag_sys
            fits_df_list.append(fit_df)

        self.init_lc_fits = Lightcurves(pd.concat(fits_df_list))

    def _get_init_rest_lightcurves(self, bands=None):
        """Obtains the corrected rest-frame light curves from the initial
        light-curve fits.

        Returns
        -------
        bands : list-like, default ``None``
            Bands for which to extract the light curves. All bands are used by default.
        """
        # always add Bessell_B band
        if bands is None:
            bands = self.lcs.bands
        if 'Bessell_B' not in bands:
            bands = ['Bessell_B'] + bands
        self.init_lc_parameters = {'tmax':None, 'tmax_err':None, 
                                   'mmax':None, 'mmax_err':None, 
                                   'dm15':None, 'dm15_err':None, 
                                   'colour':None, 'colour_err':None, 
                                   }

        rest_df_list = []
        phases = (self.times_pred - self.init_tmax) / (1 + self.z)
        mask = (-50 <= phases) & (phases <= 50)  # mask for quicker calculation
        for band in bands:
            rest_eff_wave = self.filters[band].eff_wave * (1 + self.z)
            wavelengths_pred = np.zeros_like(self.times_pred) + rest_eff_wave            
            mu, cov = self.init_gp_predict(self.times_pred[mask], wavelengths_pred[mask], return_cov=True)

            # correct for MW dust extinction and redshift 
            A = self.filters[band].calculate_extinction(self.ra, self.dec)
            correction = 10**(0.4*A) * (1 + self.z)
            mu *= correction
            cov *= correction ** 2
            std = np.sqrt(np.diag(cov))

            self.init_lc_parameters.update({band:{}})
            # calculate light-curve parameters
            _, tmax, tmax_err, fmax, fmax_err, df15, df15_err = self._calculate_lc_params(phases[mask], mu, cov)
            # tmax - this is actually phase so the inital tmax is added
            self.init_lc_parameters[band]['tmax'] = np.round(tmax + self.init_tmax, 3)
            self.init_lc_parameters[band]['tmax_err'] = np.round(tmax_err, 3)
            # mmax
            zp = self.filters[band].calculate_zp(self.filters[band].mag_sys)
            mmax, mmax_err = flux_to_mag(fmax, fmax_err, zp)
            self.init_lc_parameters[band]['mmax'] = np.round(mmax, 3)
            self.init_lc_parameters[band]['mmax_err'] = np.round(mmax_err, 3)
            # dm15 
            dm15, dm15_err = flux_to_mag(df15, df15_err, 0.0)
            self.init_lc_parameters[band]['dm15'] = np.round(dm15, 3)
            self.init_lc_parameters[band]['dm15_err'] = np.round(dm15_err, 3)

            if band == 'Bessell_B':
                # tmax - taken from the initial light-curve fit
                self.init_lc_parameters['tmax'] = self.init_tmax.copy()
                self.init_lc_parameters['tmax_err'] = self.init_tmax_err.copy()
                self.init_lc_parameters[band]['tmax'] = self.init_tmax.copy()
                self.init_lc_parameters[band]['tmax_err'] = self.init_tmax_err.copy()
                self.init_lc_parameters['mmax'] = self.init_lc_parameters[band]['mmax']
                self.init_lc_parameters['mmax_err'] = self.init_lc_parameters[band]['mmax_err']
                self.init_lc_parameters['dm15'] = self.init_lc_parameters[band]['dm15']
                self.init_lc_parameters['dm15_err'] = self.init_lc_parameters[band]['dm15_err']

                # GP prediction
                band2 = 'Bessell_V'  # for (B-V)
                zp2 = self.filters[band2].calculate_zp(self.filters[band2].mag_sys)
                rest_eff_wave2 = self.filters[band2].eff_wave * (1 + self.z)
                wavelengths_pred = np.array([rest_eff_wave, rest_eff_wave2])
                times_pred = np.zeros_like(wavelengths_pred) + self.init_tmax
                mu2, cov2 = self.init_gp_predict(times_pred, wavelengths_pred, return_cov=True)

                # propagate MW dust extinction correction and redshift
                A2 = self.filters[band2].calculate_extinction(self.ra, self.dec)
                corr_array = np.array([10 ** (0.4 * A), 10 ** (0.4 * A2)]) * (1 + self.z)
                mu2 *= corr_array
                cov2 *= np.outer(corr_array, corr_array)
                std2 = np.sqrt(np.diag(cov2))

                # calculate colour
                flux_ratio = mu2[0] / mu2[1]
                flux_ratio_error = np.abs(flux_ratio) * np.sqrt((std2[0] / mu2[0]) ** 2 + (std2[1] / mu2[1]) ** 2 - 2 * (cov2[0][1] / (mu2[0] * mu2[1])))
                colour, colour_err = flux_to_mag(flux_ratio, flux_ratio_error, zp=(zp-zp2))
                self.init_lc_parameters['colour'] = np.round(colour, 3)
                self.init_lc_parameters['colour_err'] = np.round(colour_err, 3)

            # store rest-frame light-curves
            rest_df = pd.DataFrame({"time": self.times_pred[mask], "flux": mu, "flux_err": std})
            rest_df["zp"] = zp
            rest_df["band"] = band
            rest_df["mag_sys"] = self.filters[band].mag_sys
            rest_df_list.append(rest_df)

        self.init_rest_lcs = Lightcurves(pd.concat(rest_df_list))

    def get_obs_params(self, bands=None):
        """Obtains the time and magnitude of maximum for the observed light curves.

        No corrections are applied at all.

        Returns
        -------
        bands : list-like, default ``None``
            Bands for which to extract the parameters.
        """
        if bands is None:
            bands = self.bands

        phases = (self.times_pred - self.init_tmax) / (1 + self.z)
        self.obs_parameters = {band:{} for band in bands}
        for band in bands:
            obs_eff_wave = self.filters[band].eff_wave
            wavelengths_pred = np.zeros_like(self.times_pred) + obs_eff_wave
            mask = (-15 <= phases) & (phases <= 10)  # mask for quicker calculation
            mu, cov = self.init_gp_predict(self.times_pred[mask], wavelengths_pred[mask], return_cov=True)

            # calculate light-curve parameters
            _, tmax, tmax_err, fmax, fmax_err, _, _ = self._calculate_lc_params(self.times_pred[mask], mu, cov)
            # tmax - taken from the initial light-curve fit
            self.obs_parameters[band]['tmax'] = np.round(tmax, 3)
            self.obs_parameters[band]['tmax_err'] = np.round(tmax_err, 3)
            # mmax
            zp = self.filters[band].calculate_zp(self.filters[band].mag_sys)
            mmax, mmax_err = flux_to_mag(fmax, fmax_err, zp)
            self.obs_parameters[band]['mmax'] = np.round(mmax, 3)
            self.obs_parameters[band]['mmax_err'] = np.round(mmax_err, 3)

    ########################
    # Magnling surface fit #
    ########################
    def gp_predict(self, times_pred, wavelengths_pred, return_cov=True):
        """Returns the Gaussian Process prediction of the mangling surface fit.

        Parameters
        ----------
        times_pred : ndarray
            Time-axis grid.
        wavelengths_pred : ndarray
            Wavelength-axis grid.
        return_cov : bool, optional
            Whether to return the covariance or variance.

        Returns
        -------
        mu: ndarray
            Mean of the Gaussian Process.
        var or cov: ndarray
            Either the variance or covariance of the Gaussian Process.
        """
        X_test, y, _, y_norm = prepare_gp_inputs(times_pred, wavelengths_pred, 
                                                self.stacked_ratios, self.stacked_errors, 
                                                fit_log=False,
                                                wave_log=self.wave_log)
        # GP prediction
        if return_cov is True:
            mu, cov = self.gp_model.predict(y, X_test=X_test, return_cov=True)
            # renormalise outputs
            mu *= y_norm
            cov *= y_norm ** 2
            return mu, cov
        else:
            mu, var = self.gp_model.predict(y, X_test=X_test, return_var=True)
            # renormalise outputs
            mu *= y_norm
            var *= y_norm ** 2
            return mu, var

    def fit(self, bands=None, k1='Matern52', fit_log=False, wave_log=False, skip_lcs_fit=False):
        """Fits and corrects the multi-colour light curves with an SED template.

        The corrections include Milky-Way dust extinction and mangling of the SED 
        (i.e. K-correction). Rest-frame light curves and parameters are calculated as end products.

        Parameters
        ----------
        bands : list-like, default ``None``
            Bands used for fitting light curves. By default, use all the available bands.
        k1: str
            Kernel to be used for the time axis. Either ``Matern52``,
            ``Matern32`` or ``ExpSquared``.
        fit_log: bool, default ``False``.
            Whether to initiallly fit the light curves in logarithmic (base 10) scale.
            The mangling is always done in flux space.
        wave_log: bool, default ``False``.
            Whether to use logarithmic (base 10) scale for the 
            wavelength axis.
        skip_lcs_fit: bool, default ``False``.
            Whether to skip the initial light-curve fit.
        """
        if skip_lcs_fit is not True:
            # GP fit to the light curve to get initial tmax
            self.fit_lcs(bands, k1=k1, fit_log=fit_log, wave_log=wave_log)  
        self.k1 = k1
        self.wave_log = wave_log

        ########################
        # Get SED light curves #
        ########################
        # get rest-frame phase and wavelength coverage to mask the SED
        # phase - use time coverage from initial light-curve fit
        fits = self.init_lc_fits[self.init_lc_fits.bands[0]]
        rest_phases = (fits.times - self.init_tmax) / (1 + self.z)
        min_phase, max_phase = np.min(rest_phases), np.max(rest_phases)
        # wavelength - coverage is extended 10% on both edges
        wavelengths = np.hstack([self.filters[band].wavelength for band in self.bands]) 
        rest_wavelengths = wavelengths / (1 + self.z)
        min_wave, max_wave = np.min(rest_wavelengths), np.max(rest_wavelengths)
        min_wave -= 0.1 * min_wave
        max_wave += 0.1 * max_wave
        # mask
        self.sed.mask_sed(min_phase, max_phase, min_wave , max_wave)

        # SED initial rest-frame light curves
        self.sed.calculate_rest_lightcurves(self.filters)
        self.sed.init_rest_lcs = self.sed.rest_lcs.copy()
        # SED is moved to observer frame and extinction is applied
        # getting initial observer-frame light curves
        self.sed.calculate_obs_lightcurves(self.filters, self.bands)
        # use the interpolated SED light curves to match the time of observations
        sed_lcs = self.sed.obs_lcs.copy()  # this is a DataFrame
        sed_times = sed_lcs.phase.values * (1 + self.z) + self.init_tmax

        #########################
        # Calculate flux ratios #
        #########################
        fitting_bands = bands
        if bands is None:
            fitting_bands = self.bands            

        times, wavelenths = [], []
        flux_ratios, flux_errors = [], []
        for band in fitting_bands:
            # this mask avoids observations beyond the SED template limits
            mask = (self.lcs[band].times >= sed_times.min()) & (self.lcs[band].times <= sed_times.max())
            self.lcs[band].mask_lc(mask, copy=True)
            sed_flux = np.interp(self.lcs[band].masked_times, sed_times, sed_lcs[band].values,
            )

            times.append(self.lcs[band].masked_times)
            wavelenths.append(
                [self.filters[band].eff_wave] * len(self.lcs[band].masked_times)
            )
            flux_ratios.append(self.lcs[band].masked_fluxes / sed_flux)
            flux_errors.append(self.lcs[band].masked_flux_errors / sed_flux)

        # prepare data for 2D fit
        self.stacked_times = np.hstack(times)
        self.stacked_wavelengths = np.hstack(wavelenths)
        self.stacked_ratios = np.hstack(flux_ratios)
        self.stacked_errors = np.hstack(flux_errors)

        ############################
        # Mangling surface fitting #
        ############################
        gp_model = fit_gp_model(self.stacked_times, self.stacked_wavelengths, 
                                self.stacked_ratios, self.stacked_errors, k1=k1, fit_mean=True, fit_log=False,
                                wave_log=wave_log)
        self.gp_model = gp_model  # store GP model

        # light-curve prediction for visualisation purposes
        fits_df_list = []
        for band in fitting_bands:
            eff_wave = self.filters[band]["eff_wave"]
            # 'self.times_pred' was stored from 'self.fit_lcs()'
            wavelengths_pred = np.zeros_like(self.times_pred) + eff_wave
            mu, var = self.gp_predict(self.times_pred, wavelengths_pred, return_cov=False)
            std = np.sqrt(var)

            # get SED light curve, in the given band, and "mangle" it
            sed_flux = np.interp(
                self.times_pred, sed_times, sed_lcs[band].values, left=0.0, right=0.0
            )
            lc_fit = sed_flux * mu  # mangling
            lc_std = sed_flux * std

            # store light-curve fits
            fit_df = pd.DataFrame({"time": self.times_pred, "flux": lc_fit, "flux_err": lc_std})
            fit_df["zp"] = self.lcs[band].zp
            fit_df["band"] = band
            fit_df["mag_sys"] = self.lcs[band].mag_sys
            fits_df_list.append(fit_df)

        self.lc_fits = Lightcurves(pd.concat(fits_df_list))

        self._mangle_sed()
        self._get_rest_lightcurves()
        self._extract_lc_params()

    def _mangle_sed(self):
        """Mangles the supernova SED."""
        mangled_sed = {"flux": [], "flux_err": []}
        for phase in np.unique(self.sed.phase):
            ##########################
            # SED at the given phase #
            ##########################
            phase_mask = self.sed.phase == phase
            sed_wave = self.sed.wave[phase_mask]
            sed_flux = self.sed.flux[phase_mask]

            #####################
            # Mangling function #
            #####################
            phases_pred = np.zeros_like(sed_wave) + phase
            times_pred = phases_pred * (1 + self.z) + self.init_tmax
            mu, var = self.gp_predict(times_pred, sed_wave, return_cov=False)
            std = np.sqrt(var)

            ##########################################
            # SED convolution with mangling function #
            ##########################################
            mangled_flux = mu * sed_flux
            mangled_flux_err = std * sed_flux

            mangled_sed["flux"] += list(mangled_flux)
            mangled_sed["flux_err"] += list(mangled_flux_err)

        self.sed.flux = np.array(mangled_sed["flux"])
        self.sed.flux_err = np.array(mangled_sed["flux_err"])
        self.sed.mangled = True

    def _get_rest_lightcurves(self):
        """Calculates the rest-frame light curves after corrections."""
        self.sed.calculate_rest_lightcurves(self.filters)
        lcs_df_list = []
        for band in self.filters.bands:
            if band not in self.sed.rest_lcs:
                # the SED does not cover this band -> skip it
                continue
            mag_sys = self.filters[band].mag_sys
            zp = self.filters[band].calculate_zp(mag_sys)

            # rest-frame light curve
            lc = self.sed.rest_lcs
            lc_df = pd.DataFrame(
                {
                    "time": lc.phase.values,
                    "flux": lc[band].values,
                    "flux_err": lc[f"{band}_err"].values,
                }
            )
            lc_df["zp"] = zp
            lc_df["band"] = band
            lc_df["mag_sys"] = mag_sys
            lcs_df_list.append(lc_df)

        self.rest_lcs = Lightcurves(pd.concat(lcs_df_list))

    def _calculate_lc_params(self, times_pred, mu, cov):

        ######################
        # Estimate peak time #
        ######################
        # get epoch at maximum - monte-carlo sampling
        lc_sample = np.random.multivariate_normal(mu, cov, size=5000)
        tmax_list = []
        for mangled_fluxes in lc_sample:
            peak_ids = peak.indexes(mangled_fluxes, thres=0.3, min_dist=len(times_pred) // 2)
            if len(peak_ids) == 0:
                # if no peak is found, just use the maximum
                max_id = np.argmax(mangled_fluxes)
            else:
                max_id = peak_ids[0]
            tmax_list.append(times_pred[max_id])
        # store time of maximum
        tmax = np.nanmean(tmax_list)
        tmax = np.round(tmax, 3)
        # add uncertainty floor (time step of 0.1 days)
        tmax_err = np.sqrt(np.nanstd(tmax_list) ** 2 + 0.1 ** 2)
        tmax_err = np.round(tmax_err, 3) 
        
        ######################
        # Estimate peak flux #
        ######################
        peak_ids = peak.indexes(mu, thres=0.3, min_dist=len(times_pred) // 2)
        if len(peak_ids) == 0:
            # if no peak is found, just use the maximum
            max_id = np.argmax(mu)
            print('WARNING: no peak found - using the maximum instead')
        else:
            max_id = peak_ids[0]
        # store flux at maximum
        fmax = mu[max_id]
        std = np.sqrt(np.diag(cov))
        fmax_err = std[max_id]

        ####################
        # Estimate Stretch #
        ####################
        t15 = times_pred[max_id] + 15
        ind15 = np.argmin(np.abs(times_pred - t15))
        f15 = mu[ind15]
        f15_err = std[ind15]
        df15 = f15/fmax
        # error propagation
        f15_cov = cov[max_id][ind15]  # covariance, in flux, at peak and 15 days
        df15_err = np.abs(df15) * np.sqrt((f15_err / f15) ** 2 + (fmax_err / fmax) ** 2 - 2 * (f15_cov / (f15 * fmax)))

        return max_id, tmax, tmax_err, fmax, fmax_err, df15, df15_err

    def _extract_lc_params(self):
        #####################
        # Get GP prediction #
        #####################
        # The prediction arrays are in observer-frame as the GP fit
        # was made in observer-frame
        band = 'Bessell_B'
        phases = self.rest_lcs[band].times
        mask = (-10 <= phases) & (phases <= 40)

        times_pred = phases[mask] * (1 + self.z) + self.init_tmax
        rest_eff_wave = self.filters[band].eff_wave * (1 + self.z)
        wavelengths_pred = np.zeros_like(times_pred) + rest_eff_wave
        _, cov = self.gp_predict(times_pred, wavelengths_pred, return_cov=True)

        # match the diagonal of the covariance matrix from the mangling surface
        # to the rest-frame light-curve errors 
        norm = self.rest_lcs[band].flux_errors[mask] / np.sqrt(np.diag(cov))
        cov *= np.outer(norm, norm)
        fluxes = self.rest_lcs[band].fluxes[mask]

        # calculate light-curve parameters
        max_id, tmax, tmax_err, fmax, fmax_err, df15, df15_err = self._calculate_lc_params(times_pred, fluxes, cov)
        # tmax
        self.tmax = np.round(tmax, 3)
        self.tmax_err = np.round(np.sqrt(tmax_err ** 2 + 0.1 ** 2), 3)
        # mmax
        mmax, mmax_err = flux_to_mag(fmax, fmax_err, self.rest_lcs[band].zp)
        self.mmax = np.round(mmax, 3)
        # add uncertainty floor (0.01 mag)
        self.mmax_err = np.round(np.sqrt(mmax_err ** 2 + 0.01 ** 2), 3)
        # dm15
        dm15, dm15_err = flux_to_mag(df15, df15_err, 0.0)
        self.dm15 = np.round(dm15, 3)
        # add uncertainty floor (0.01 mag)
        self.dm15_err = np.round(np.sqrt(dm15_err ** 2 + 0.01 ** 2), 3)

        ###################
        # Estimate Colour #
        ###################
        # get GP prediction
        band2 = 'Bessell_V'  # for (B-V)
        rest_eff_wave2 = self.filters[band2].eff_wave * (1 + self.z)
        wavelengths_pred = np.array([rest_eff_wave, rest_eff_wave2])
        times_pred = np.zeros_like(wavelengths_pred) + tmax
        _, cov = self.gp_predict(times_pred, wavelengths_pred, return_cov=True)

        # scale the covariance (2 x 2 matrix) to match the light-curve uncertainties
        errors = np.array([self.rest_lcs[band].flux_errors[mask][max_id], 
                           self.rest_lcs[band2].flux_errors[mask][max_id]])
        norm = errors / np.sqrt(np.diag(cov))
        cov *= np.outer(norm, norm)

        # calculate colour
        mB = self.rest_lcs[band].magnitudes[mask][max_id]
        mV = self.rest_lcs[band2].magnitudes[mask][max_id]
        colour = (mB - mV)
        self.colour = np.round(colour, 3)
        # propagate errors
        mB_err = self.rest_lcs[band].mag_errors[mask][max_id]
        mV_err = self.rest_lcs[band2].mag_errors[mask][max_id]
        flux_colour_cov = cov[0][1]
        # propagate covariance from flux to magnitude
        fB, _ = mag_to_flux(mB, zp=self.rest_lcs[band].zp)
        fV, _ = mag_to_flux(mV, zp=self.rest_lcs[band2].zp)
        colour_cov = ((2.5 ** 2) * flux_colour_cov) / (np.log(10) ** 2 * fB * fV)
        # add uncertainty floor (0.01 mag)
        colour_err = np.sqrt(mB_err ** 2 + mV_err ** 2 - 2 * colour_cov + 0.01 ** 2)
        self.colour_err = np.round(colour_err, 3)

        # store all parameters together
        self.lc_parameters = {
            "tmax": self.tmax,
            "tmax_err": self.tmax_err,
            "Bmax": self.mmax,
            "Bmax_err": self.mmax_err,
            "dm15": self.dm15,
            "dm15_err": self.dm15_err,
            "colour": self.colour,
            "colour_err": self.colour_err,
        }

    # Extra Functions
    # this are not essential but help assessing the fits
    # and output results, among other things
    def plot_lcs(self, plot_mag=False, fig_name=None):
        """Plots the observed light-curves.

        Parameters
        ----------
        plot_mag : bool, default ``False``
            If ``True``, plots the bands in magnitude space.
        fig_name : str, default ``None``
            If  given, name of the output plot.
        """
        palette1 = [plt.get_cmap("Dark2")(i) for i in np.arange(8)]
        palette2 = [plt.get_cmap("Set1")(i) for i in np.arange(9)]
        palette3 = [plt.get_cmap("Pastel2")(i) for i in np.arange(8)]
        colours = palette1 + palette2 + palette3 + palette1 + palette2

        # shift in time for visualization purposes
        min_time = self.lcs[self.bands[1]].times.min()
        tmax_str = str(min_time.astype(int))
        zeros = "0" * len(tmax_str[2:])
        t_offset = int(tmax_str[:2] + zeros)

        ZP = 27.5  # global zero-point for visualization

        h = 3  # columns
        v = math.ceil(len(self.bands) / h)  # rows

        fig = plt.figure(figsize=(15, 5 * v))
        gs = gridspec.GridSpec(v, h)

        for i, band in enumerate(self.bands):
            j = math.ceil(i % h)
            k = i // h
            ax = plt.subplot(gs[k, j])

            x = self.lcs[band].times - t_offset
            if plot_mag is False:
                y_norm = change_zp(1.0, self.lcs[band].zp, ZP)
                y = self.lcs[band].fluxes * y_norm
                yerr = self.lcs[band].flux_errors * y_norm
            else:
                y = self.lcs[band].magnitudes
                yerr = self.lcs[band].mag_errors
                ax.invert_yaxis()

            colour = colours[i]
            data_par = dict(
                fmt="o",
                color=colour,
                capsize=3,
                capthick=2,
                ms=8,
                elinewidth=3,
                mec="k",
            )

            # light curves
            ax.errorbar(x, y, yerr, label=band, **data_par)

            ax.xaxis.set_tick_params(labelsize=15)
            ax.yaxis.set_tick_params(labelsize=15)
            ax.minorticks_on()
            ax.tick_params(
                which="major", length=6, width=1, direction="in", top=True, right=True
            )
            ax.tick_params(
                which="minor", length=3, width=1, direction="in", top=True, right=True
            )
            ax.legend(fontsize=16)

        fig.text(
            0.5,
            0.92,
            f"{self.name} (z = {self.z:.5})",
            ha="center",
            fontsize=20,
            family="serif",
        )
        fig.text(
            0.5,
            0.05,
            f"Time - {t_offset} [days]",
            ha="center",
            fontsize=18,
            family="serif",
        )
        if not plot_mag:
            fig.text(
                0.05,
                0.5,
                f"Flux (ZP = {ZP})",
                va="center",
                rotation="vertical",
                fontsize=18,
                family="serif",
            )
        else:
            fig.text(
                0.05,
                0.5,
                r"Apparent Magnitude",
                va="center",
                rotation="vertical",
                fontsize=18,
                family="serif",
            )

        if fig_name is not None:
            plt.savefig(fig_name)

        plt.show()

    def plot_fits(self, plot_mag=False, init_fits=False, fig_name=None):
        """Plots the light-curves fits results.

        Plots the observed data for each band together with the gaussian process fits.
        The time of :math:`B`-band peak is shown as a vertical dashed line.

        Parameters
        ----------
        plot_mag : bool, default ``False``
            If ``True``, plots the bands in magnitude space.
        init_fits: bool, default ``False``
            If ``True``, plots the initial light-curve fits.
        fig_name : str, default ``None``
            If  given, name of the output plot.
        """
        palette1 = [plt.get_cmap("Dark2")(i) for i in np.arange(8)]
        palette2 = [plt.get_cmap("Set1")(i) for i in np.arange(9)]
        palette3 = [plt.get_cmap("Pastel2")(i) for i in np.arange(8)]
        colours = palette1 + palette2 + palette3 + palette1 + palette2

        # shift in time for visualization purposes
        try:
            tmax_str = str(self.tmax.astype(int))
        except:
            tmax_str = str(self.init_tmax.astype(int))
        zeros = "0" * len(tmax_str[2:])
        t_offset = int(tmax_str[:2] + zeros)

        # data and fits
        data = self.lcs
        try:
            fits = self.lc_fits
            tmax = self.tmax
        except:
            fits = self.init_lc_fits
            tmax = self.init_tmax
        finally:
            if init_fits is True:
                fits = self.init_lc_fits
                tmax = self.init_tmax

        ZP = 27.5  # global zero-point for visualization

        h = 3  # columns
        v = math.ceil(len(fits.bands) / h)  # rows

        fig = plt.figure(figsize=(15, 5 * v))
        gs = gridspec.GridSpec(v * 2, h, height_ratios=[3, 1] * v)

        for i, band in enumerate(fits.bands):
            j = math.ceil(i % h)
            k = i // h * 2
            ax = plt.subplot(gs[k, j])
            ax2 = plt.subplot(gs[k + 1, j])

            x = data[band].times - t_offset
            x_fit = fits[band].times - t_offset

            if plot_mag is False:
                y_norm = change_zp(1.0, data[band].zp, ZP)
                y = data[band].fluxes * y_norm
                yerr = data[band].flux_errors * y_norm
                y_fit = fits[band].fluxes * y_norm
                yerr_fit = fits[band].flux_errors * y_norm
            else:
                y = data[band].magnitudes
                yerr = data[band].mag_errors
                y_fit = fits[band].magnitudes
                yerr_fit = fits[band].mag_errors
                ax.invert_yaxis()
                ax2.invert_yaxis()

            colour = colours[i]
            data_par = dict(
                fmt="o",
                color=colour,
                capsize=3,
                capthick=2,
                ms=8,
                elinewidth=3,
                mec="k",
            )
            fit_par = dict(ls="-", lw=2, zorder=16, color=colour)
            fit_err_par = dict(alpha=0.3, color=colour)

            # light curves
            ax.errorbar(x, y, yerr, label=band, **data_par)
            ymin, ymax = ax.get_ylim()
            ax.plot(x_fit, y_fit, **fit_par)
            ax.fill_between(x_fit, y_fit - yerr_fit, y_fit + yerr_fit, **fit_err_par)
            ax.set_ylim(ymin, ymax)  # fits can sometimes explode

            # residuals
            res = y - np.interp(x, x_fit, y_fit)
            ax2.errorbar(x, res, yerr, **data_par)
            ymin, ymax = ax2.get_ylim()
            ax2.plot(x_fit, np.zeros_like(y_fit), **fit_par)
            ax2.fill_between(x_fit, -yerr_fit, yerr_fit, **fit_err_par)
            lim = np.max([np.abs(ymin), ymax])
            ax2.set_ylim(-lim, lim)  # symmetric limits

            for axis in [ax, ax2]:
                axis.axvline(
                    x=tmax - t_offset, color="k", linestyle="--", alpha=0.4
                )
                axis.xaxis.set_tick_params(labelsize=15)
                axis.yaxis.set_tick_params(labelsize=15)
                axis.minorticks_on()
                axis.tick_params(
                    which="major",
                    length=6,
                    width=1,
                    direction="in",
                    top=True,
                    right=True,
                )
                axis.tick_params(
                    which="minor",
                    length=3,
                    width=1,
                    direction="in",
                    top=True,
                    right=True,
                )
            ax.set_xticks([])
            ax.legend(fontsize=16)

            ax.set_xlim(x_fit.min(), x_fit.max())
            ax2.set_xlim(x_fit.min(), x_fit.max())

        fig.text(
            0.5,
            0.92,
            f"{self.name} (z = {self.z:.5})",
            ha="center",
            fontsize=20,
            family="serif",
        )
        fig.text(
            0.5,
            0.05,
            f"Time - {t_offset} [days]",
            ha="center",
            fontsize=18,
            family="serif",
        )
        if not plot_mag:
            fig.text(
                0.05,
                0.5,
                f"Flux (ZP = {ZP})",
                va="center",
                rotation="vertical",
                fontsize=18,
                family="serif",
            )
        else:
            fig.text(
                0.05,
                0.5,
                r"Apparent Magnitude",
                va="center",
                rotation="vertical",
                fontsize=18,
                family="serif",
            )

        if fig_name is not None:
            plt.savefig(fig_name)

        plt.show()

    def export_fits(self, output_file=None):
        """Exports the light-curve fits into an output file.

        Parameters
        ----------
        output_file : str, default ``None``
            Name of the output file. If ``None``, the output
            file is of the form ``<SN name>_fits.dat``.
        """
        if output_file is None:
            output_file = f"{self.name}_fits.dat"

        df_list = []
        columns = ["times", "fluxes", "flux_errors", "magnitudes", "mag_errors", "zp", "band"]

        for band in self.bands:
            band_info = self.lc_fits[band]
            band_dict = {key: band_info[key] for key in columns}
            band_df = pd.DataFrame(band_dict)

            rounding_dict = {key: 3 for key in ["times", "magnitudes", "mag_errors", "zp"]}
            band_df = band_df.round(rounding_dict)
            df_list.append(band_df[columns])

        df_fits = pd.concat(df_list)
        df_fits.to_csv(output_file, sep="\t", index=False)

    def export_restframe_lcs(self, output_file=None):
        """Exports the corrected, rest-frame light-curves into an output file.

        Parameters
        ----------
        output_file : str, default ``None``
            Name of the output file. If ``None``, the output
        file is of the form ``<SN name>_restframe_lcs.dat``.
        """

        if output_file is None:
            output_file = f"{self.name}_restframe_lcs.dat"

        df_list = []
        columns = ["times", "fluxes", "flux_errors", "magnitudes", "mag_errors", "zp", "band"]

        for band in self.rest_lcs.bands:
            band_info = self.rest_lcs[band]
            band_dict = {key: band_info[key] for key in columns}
            band_df = pd.DataFrame(band_dict)

            rounding_dict = {key: 3 for key in ["times", "magnitudes", "mag_errors", "zp"]}
            band_df = band_df.round(rounding_dict)
            df_list.append(band_df[columns])

        df_fits = pd.concat(df_list)
        df_fits.to_csv(output_file, sep="\t", index=False)
