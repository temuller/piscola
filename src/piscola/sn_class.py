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
        lc_file, nrows=1, sep='\\s+', converters=converters
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

    return sn_obj


class Supernova(object):
    """Class representing a supernova object."""

    def __init__(
        self, name, z=0.0, ra=None, dec=None, lc_file=None
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
        """
        self.name = name
        self.z = z
        self.ra = ra
        self.dec = dec

        if not self.ra or not self.dec:
            print("Warning, RA and/or DEC not specified.")

        # add light curves and filters
        if lc_file:
            lcs_df = pd.read_csv(lc_file, sep='\\s+', skiprows=2)
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
        if path is None:
            path = ""

        outfile = os.path.join(path, f"{self.name}.pisco")
        with bz2.BZ2File(outfile, "wb") as pfile:
            protocol = 4
            pickle.dump(self, pfile, protocol)

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

    def gp_predict(self, times_pred, wavelengths_pred, return_cov=True):
        """Returns the Gaussian Process prediction of the light-curve fits.

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
        cov: ndarray
            Either the variance or covariance of the Gaussian Process.
        """
        X_test, y, _, y_norm = prepare_gp_inputs(times_pred, wavelengths_pred, 
                                                self._stacked_fluxes, self._stacked_flux_errors, 
                                                fit_type=self.fit_type,
                                                wave_log=self.wave_log)

        # Function to propagate the covariance matrix through nonlinear functions
        def transform_covariance(cov_matrix, x, fit_type):
            # Compute the Jacobian at point x
            if fit_type == "log":
                jac = np.diag((10 ** x) * np.log(10))
            elif fit_type == "arcsinh":    
                jac = np.diag(1 / np.sqrt(x**2 + 1))
            # Transform the covariance matrix using the Jacobian
            transformed_cov_matrix = jac @ cov_matrix @ jac.T
            return transformed_cov_matrix

        # GP prediction
        if return_cov is True:
            return_var = False
        else:
            return_var = True

        # if return_cov and return_var are both True, the covariance flag is ignored
        if self.fit_type == "flux":
            mu, cov = self.gp_model.predict(y, X_test=X_test, return_var=return_var, return_cov=True)
        elif self.fit_type == "log":
            log_mu, log_cov = self.gp_model.predict(y, X_test=X_test, return_var=return_var, return_cov=True)
            mu = 10 ** log_mu
            cov = transform_covariance(log_cov, log_mu, self.fit_type)
        elif self.fit_type == "arcsinh":
            arcsinh_mu, arcsinh_cov = self.gp_model.predict(y, X_test=X_test, return_var=return_var, return_cov=True)
            mu = np.sinh(arcsinh_mu)
            cov = transform_covariance(arcsinh_cov, arcsinh_mu, self.fit_type)

        # renormalise outputs
        mu *= y_norm
        cov *= y_norm ** 2

        return mu, cov
        
    def fit(self, bands=None, k1='Matern52', fit_type='flux', wave_log=True, time_scale=None, wave_scale=None):
        """Fits the observed multi-colour light-curve data with Gaussian Process.

        The time of rest-frame B-band peak luminosity is estimated by finding where the derivative is equal to zero.

        Parameters
        ----------
        bands : list-like, default ``None``
            Bands used for fitting light curves. By default, use all the available bands.
        k1: str
            Kernel to be used for the time axis. Either ``Matern52``,
            ``Matern32`` or ``ExpSquared``.
        fit_type: str, default ``flux``.
            Transformation used for the light-curve fits: ``flux``, ``log``, ``arcsinh``.
        wave_log: bool, default ``True``.
            Whether to use logarithmic (base 10) scale for the 
            wavelength axis.
        time_scale: float, default ``None``
            If given, the time scale is fixed using this value, in units of days.
        wave_scale: float, default ``None``
            If given, the wavelength scale is fixed using this value, in units of angstroms.
            Note that if 'wave_log=True', the logarithm base 10 of this value is used.
        """
        ##########
        # GP fit #
        ##########
        self._stack_lcs(bands)
        gp_model = fit_gp_model(self._stacked_times, self._stacked_wavelengths, 
                                self._stacked_fluxes, self._stacked_flux_errors, k1=k1, 
                                fit_type=fit_type, wave_log=wave_log, time_scale=time_scale, 
                                wave_scale=wave_scale)
        self.gp_model = gp_model  # store GP model
        self.k1 = k1
        self.fit_type = fit_type
        self.wave_log = wave_log

        ########################
        # Estimate B-band Peak #
        ########################
        obs_eff_wave = self.filters.Bessell_B.eff_wave * (1 + self.z)
        dt = 0.1 * (1 + self.z)  # 0.1 days in rest-frame, moved to observer frame
        times_pred = np.arange(self._stacked_times.min() - 5, 
                               self._stacked_times.max() + 5,
                               dt
                               )
        self.times_pred = times_pred
        wavelengths_pred = np.zeros_like(times_pred) + obs_eff_wave
        mu, cov = self.gp_predict(times_pred, wavelengths_pred, return_cov=True)

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
        self.tmax = np.round(np.nanmean(tmax_list), 3)
        self.tmax_err = np.round(np.nanstd(tmax_list), 3)

        ############################
        # Light-curves predictions #
        ############################
        fitting_bands = bands
        if bands is None:
            fitting_bands = self.bands
            
        fits_df_list = []
        for band in fitting_bands:
            eff_wave = self.filters[band]["eff_wave"]
            wavelengths_pred = np.zeros_like(times_pred) + eff_wave
            mu, var = self.gp_predict(times_pred, wavelengths_pred, return_cov=False)
            std = np.sqrt(var)

            # store light-curve fits
            fit_df = pd.DataFrame({"time": times_pred, "flux": mu, "flux_err": std})
            fit_df["zp"] = self.lcs[band].zp
            fit_df["band"] = band
            fit_df["mag_sys"] = self.lcs[band].mag_sys
            fits_df_list.append(fit_df)

        self.lc_fits = Lightcurves(pd.concat(fits_df_list))

        # get rest-frame light curve and light-curve parameters
        self._get_rest_lightcurves()

    def _get_rest_lightcurves(self, bands=None):
        """Obtains the corrected rest-frame light curves from the fits.

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
        self.lc_parameters = {'tmax':None, 'tmax_err':None, 
                                'mmax':None, 'mmax_err':None, 
                                'dm15':None, 'dm15_err':None, 
                                'colour':None, 'colour_err':None, 
                                }

        rest_df_list = []
        phases = (self.times_pred - self.tmax) / (1 + self.z)
        mask = (-50 <= phases) & (phases <= 50)  # mask for quicker calculation
        phases = phases[mask]
        for band in bands:
            rest_eff_wave = self.filters[band].eff_wave * (1 + self.z)
            wavelengths_pred = np.zeros_like(self.times_pred) + rest_eff_wave            
            mu, cov = self.gp_predict(self.times_pred[mask], wavelengths_pred[mask], return_cov=True)

            # correct for MW dust extinction and redshift 
            A = self.filters[band].calculate_extinction(self.ra, self.dec)
            correction = 10**(0.4*A) * (1 + self.z)
            mu *= correction
            cov *= correction ** 2
            std = np.sqrt(np.diag(cov))

            self.lc_parameters.update({band:{}})
            try:
                # calculate light-curve parameters
                _, tmax, tmax_err, fmax, fmax_err, df15, df15_err = self._calculate_lc_params(phases, mu, cov)
                # tmax - this is actually phase so the inital tmax is added
                self.lc_parameters[band]['tmax'] = np.round(tmax + self.tmax, 3)
                self.lc_parameters[band]['tmax_err'] = np.round(tmax_err, 3)
                # mmax
                zp = self.filters[band].calculate_zp(self.filters[band].mag_sys)
                mmax, mmax_err = flux_to_mag(fmax, fmax_err, zp)
                self.lc_parameters[band]['mmax'] = np.round(mmax, 3)
                self.lc_parameters[band]['mmax_err'] = np.round(mmax_err, 3)
                # dm15 
                dm15, dm15_err = flux_to_mag(df15, df15_err, 0.0)
                self.lc_parameters[band]['dm15'] = np.round(dm15, 3)
                self.lc_parameters[band]['dm15_err'] = np.round(dm15_err, 3)
            except:
                self.lc_parameters[band]['tmax'] = self.lc_parameters[band]['tmax_err'] = np.nan
                self.lc_parameters[band]['mmax'] = self.lc_parameters[band]['mmax_err'] = np.nan
                self.lc_parameters[band]['dm15'] = self.lc_parameters[band]['dm15_err'] = np.nan

            if band == 'Bessell_B':
                # tmax - taken from the initial light-curve fit
                self.lc_parameters['tmax'] = self.tmax.copy()
                self.lc_parameters['tmax_err'] = self.tmax_err.copy()
                self.lc_parameters[band]['tmax'] = self.tmax.copy()
                self.lc_parameters[band]['tmax_err'] = self.tmax_err.copy()
                self.lc_parameters['mmax'] = self.lc_parameters[band]['mmax']
                self.lc_parameters['mmax_err'] = self.lc_parameters[band]['mmax_err']
                self.lc_parameters['dm15'] = self.lc_parameters[band]['dm15']
                self.lc_parameters['dm15_err'] = self.lc_parameters[band]['dm15_err']

                # GP prediction
                band2 = 'Bessell_V'  # for (B-V)
                zp2 = self.filters[band2].calculate_zp(self.filters[band2].mag_sys)
                rest_eff_wave2 = self.filters[band2].eff_wave * (1 + self.z)
                wavelengths_pred = np.array([rest_eff_wave, rest_eff_wave2])
                times_pred = np.zeros_like(wavelengths_pred) + self.tmax
                mu2, cov2 = self.gp_predict(times_pred, wavelengths_pred, return_cov=True)

                # propagate MW dust extinction correction and redshift
                A2 = self.filters[band2].calculate_extinction(self.ra, self.dec)
                corr_array = np.array([10 ** (0.4 * A), 10 ** (0.4 * A2)]) * (1 + self.z)
                mu2 *= corr_array
                cov2 *= np.outer(corr_array, corr_array)
                std2 = np.sqrt(np.diag(cov2))

                # calculate colour
                flux_ratio = mu2[0] / mu2[1]
                flux_ratio_error = np.abs(flux_ratio) * np.sqrt((std2[0] / mu2[0]) ** 2 + (std2[1] / mu2[1]) ** 2 - 2 * (cov2[0][1] / (mu2[0] * mu2[1])))
                colour, colour_err = flux_to_mag(flux_ratio, flux_ratio_error, zp=(zp - zp2))
                self.lc_parameters['colour'] = np.round(colour, 3)
                self.lc_parameters['colour_err'] = np.round(colour_err, 3)

            # store rest-frame light-curves
            rest_df = pd.DataFrame({"time": self.times_pred[mask], "flux": mu, "flux_err": std})
            rest_df["zp"] = zp
            rest_df["band"] = band
            rest_df["mag_sys"] = self.filters[band].mag_sys
            rest_df_list.append(rest_df)

        self.rest_lcs = Lightcurves(pd.concat(rest_df_list))

    def _calculate_lc_params(self, times_pred, mu, cov):
        ######################
        # Estimate peak time #
        ######################
        # get epoch at maximum - monte-carlo sampling
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
        # store time of maximum
        tmax = np.nanmean(tmax_list)
        tmax = np.round(tmax, 3)
        tmax_err = np.sqrt(np.nanstd(tmax_list) ** 2)
        tmax_err = np.round(tmax_err, 3) 
        
        ######################
        # Estimate peak flux #
        ######################
        peak_ids = peak.indexes(mu, thres=0.3, min_dist=len(times_pred) // 2)
        if len(peak_ids) == 0:
            # if no peak is found, just use the maximum
            max_id = np.argmax(mu)
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
        df15 = f15 / fmax
        # error propagation
        f15_cov = cov[max_id][ind15]  # covariance, in flux, at peak and 15 days
        df15_err = np.abs(df15) * np.sqrt((f15_err / f15) ** 2 + (fmax_err / fmax) ** 2 - 2 * (f15_cov / (f15 * fmax)))

        return max_id, tmax, tmax_err, fmax, fmax_err, df15, df15_err

    ################### 
    # Extra Functions #
    ###################
    # this are not essential but help assessing the fits
    # and output results, among other things
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

        phases = (self.times_pred - self.tmax) / (1 + self.z)
        self.obs_parameters = {band:{} for band in bands}
        for band in bands:
            obs_eff_wave = self.filters[band].eff_wave
            wavelengths_pred = np.zeros_like(self.times_pred) + obs_eff_wave
            mask = (-15 <= phases) & (phases <= 10)  # mask for quicker calculation
            mu, cov = self.gp_predict(self.times_pred[mask], wavelengths_pred[mask], return_cov=True)

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

    def plot_fits(self, plot_mag=False, fits=False, fig_name=None):
        """Plots the light-curves fits results.

        Plots the observed data for each band together with the gaussian process fits.
        The time of :math:`B`-band peak is shown as a vertical dashed line.

        Parameters
        ----------
        plot_mag : bool, default ``False``
            If ``True``, plots the bands in magnitude space.
        fits: bool, default ``False``
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
            tmax_str = str(self.tmax.astype(int))
        zeros = "0" * len(tmax_str[2:])
        t_offset = int(tmax_str[:2] + zeros)

        # data and fits
        data = self.lcs
        try:
            fits = self.lc_fits
            tmax = self.tmax
        except:
            fits = self.lc_fits
            tmax = self.tmax
        finally:
            if fits is True:
                fits = self.lc_fits
                tmax = self.tmax

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
