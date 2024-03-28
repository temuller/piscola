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

from .utils import change_zp
from .sed_class import SEDTemplate
from .filters_class import MultiFilters
from .lightcurves_class import Lightcurves
from .gaussian_process import prepare_gp_inputs, fit_gp_model


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
        lc_file, nrows=1, delim_whitespace=True, converters=converters
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
        self, name, z=0.0, ra=None, dec=None, lc_file=None, template="conley09f"
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
        template: str, default ``conley09f``
            Name of the spectral energy distribution (SED) template.
            E.g., ``conley09f``, ``jla``, etc.
        """
        self.name = name
        self.z = z
        self.ra = ra
        self.dec = dec
        self._init_fits = None
        self._fit_results = None

        if not self.ra or not self.dec:
            print("Warning, RA and/or DEC not specified.")

        # add light curves and filters
        if lc_file:
            lcs_df = pd.read_csv(lc_file, delim_whitespace=True, skiprows=2)
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
        if path is None:
            path = ""

        outfile = os.path.join(path, f"{self.name}.pisco")
        with bz2.BZ2File(outfile, "wb") as pfile:
            protocol = 4
            pickle.dump(self, pfile, protocol)

    def set_sed_template(self, template):
        """Sets the SED template to be used for the mangling function.

        Parameters
        ----------
        template : str
            Template name. E.g., ``conley09f``, ``jla``, etc.
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

            new_zp = self.filters[band].calc_zp(mag_sys)

            self.lcs[band].fluxes = change_zp(self.lcs[band].fluxes, current_zp, new_zp)
            self.lcs[band].flux_errors = change_zp(
                self.lcs[band].flux_errors, current_zp, new_zp
            )
            self.lcs[band].zp = new_zp

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

    def fit_lcs(self, bands=None, k1='Matern52', use_log=True):
        """Fits the multi-colour observed light-curve data with Gaussian Process.

        The time of rest-frame B-band peak luminosity is estimated by finding where the derivative is equal to zero.

        Parameters
        ----------
        bands : list-like, default ``None``
            Bands used for fitting light curves. By default, use all the available bands.
        k1: str
            Kernel to be used for the time axis. Either ``Matern52``,
            ``Matern32`` or ``ExpSquared``.
        use_log: bool, default ``True``.
            Whether to use logarithmic (base 10) scale for the 
            wavelength axis.
        """
        ##########
        # GP fit #
        ##########
        self._stack_lcs(bands)
        gp_model = fit_gp_model(self._stacked_times, self._stacked_wavelengths, 
                                self._stacked_fluxes, self._stacked_flux_errors, k1=k1, use_log=use_log)
        self.init_gp_model = gp_model  # store GP model
        self.init_y_norm = self._stacked_fluxes.max()  # store normalisation

        ########################
        # Estimate B-band Peak #
        ########################
        rest_eff_wave = self.filters.Bessell_B.eff_wave * (1 + self.z)
        dt = 0.1 * (1 + self.z)  # 0.1 days in rest-frame, moved to observer frame
        times_pred = np.arange(self._stacked_times.min() - 5, 
                               self._stacked_times.max() + 10,
                               dt
                               )
        # store time array for the prediction in 'self.fit()'
        self.times_pred = times_pred
        wavelengths_pred = np.zeros_like(times_pred) + rest_eff_wave
        # arrays for GP predictions
        X_test, y, _ = prepare_gp_inputs(times_pred, wavelengths_pred, 
                                         self._stacked_fluxes, self._stacked_flux_errors, 
                                         self.init_y_norm,
                                         use_log=use_log)
        # GP prediction
        mu, cov = gp_model.predict(y, X_test=X_test, return_cov=True)

        # monte-carlo sampling
        mc_lcs = np.random.multivariate_normal(mu, cov, size=5000)
        tmax_list = []
        for lc in mc_lcs:
            peak_ids = peak.indexes(lc, thres=0.3, min_dist=len(times_pred) // 2)
            if len(peak_ids) == 0:
                # if no peak is found, just use the maximum
                max_id = np.argmax(lc)
            else:
                max_id = peak_ids[0]
            tmax_list.append(times_pred[max_id])
        # save time of maximum
        self.init_tmax = np.nanmean(tmax_list)
        self.init_tmax_err = np.nanstd(tmax_list)

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
            # arrays for GP predictions
            X_test, y, _ = prepare_gp_inputs(times_pred, wavelengths_pred, 
                                             self._stacked_fluxes, self._stacked_flux_errors, 
                                             self.init_y_norm,
                                             use_log=use_log)
            # GP prediction
            mu, var = gp_model.predict(y, X_test=X_test, return_var=True)
            std = np.sqrt(var)
            # renormalise outputs
            mu *= self.init_y_norm
            std *= self.init_y_norm

            # store light-curve fits
            fit_df = pd.DataFrame({"time": times_pred, "flux": mu, "flux_err": std})
            fit_df["zp"] = self.lcs[band].zp
            fit_df["band"] = band
            fit_df["mag_sys"] = self.lcs[band].mag_sys
            fits_df_list.append(fit_df)

        self.init_lc_fits = Lightcurves(pd.concat(fits_df_list))

    def fit(self, bands=None, k1='Matern52', use_log=True):
        """Fits and corrects the multi-colour light curves.

        The corrections include Milky-Way dust extinction and mangling of the SED 
        (i.e. K-correction). Rest-frame light curves and parameters are calculated as end products.

        Parameters
        ----------
        bands : list-like, default ``None``
            Bands used for fitting light curves. By default, use all the available bands.
        k1: str
            Kernel to be used for the time axis. Either ``Matern52``,
            ``Matern32`` or ``ExpSquared``.
        use_log: bool, default ``True``.
            Whether to use logarithmic (base 10) scale for the 
            wavelength axis.
        """
        # GP fit to the light curve to get initial tmax
        self.fit_lcs(bands, k1, use_log)  
        self.k1 = k1
        self.use_log = use_log

        ########################
        # Get SED light curves #
        ########################
        # get rest-frame phase coverage to mask the SED
        fits = self.init_lc_fits[self.init_lc_fits.bands[0]]
        rest_phases = (fits.times - self.init_tmax) / (1 + self.z)
        min_phase, max_phase = np.min(rest_phases), np.max(rest_phases)
        self.sed.mask_sed(min_phase, max_phase)

        # SED observer-frame light curves
        self.sed.calculate_obs_lightcurves(self.filters)
        # interpolated SED light curves to match the time of observations
        sed_lcs = self.sed.obs_lcs_fit  # this is a DataFrame
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
            mask = self.lcs[band].times <= sed_times.max()
            self.lcs[band].mask_lc(mask, copy=True)
            sed_flux = np.interp(
                self.lcs[band].masked_times,
                sed_times,
                sed_lcs[band].values,
            )

            # prevents negative numbers
            mask = sed_flux > 0  # not the best solution, but seems to work

            times.append(self.lcs[band].masked_times[mask])
            wavelenths.append(
                [self.filters[band].eff_wave] * len(self.lcs[band].masked_times[mask])
            )
            flux_ratios.append(self.lcs[band].masked_fluxes[mask] / sed_flux[mask])
            flux_errors.append(self.lcs[band].masked_flux_errors[mask] / sed_flux[mask])

        # prepare data for 2D fit
        self.stacked_times = np.hstack(times)
        self.stacked_wavelengths = np.hstack(wavelenths)
        self.stacked_ratios = np.hstack(flux_ratios)
        self.stacked_errors = np.hstack(flux_errors)

        ############################
        # Mangling surface fitting #
        ############################
        gp_model = fit_gp_model(self.stacked_times, self.stacked_wavelengths, 
                                self.stacked_ratios, self.stacked_errors, k1=k1, use_log=use_log)
        self.gp_model = gp_model  # store GP model
        self.y_norm = self.stacked_ratios.max()  # store normalisation

        fits_df_list = []
        for band in fitting_bands:
            eff_wave = self.filters[band]["eff_wave"]
            # 'self.times_pred' was stored from 'self.fit_lcs()'
            wavelengths_pred = np.zeros_like(self.times_pred) + eff_wave
            # arrays for GP predictions
            X_test, y, _ = prepare_gp_inputs(self.times_pred, wavelengths_pred, 
                                             self.stacked_ratios, self.stacked_errors, 
                                             self.y_norm,
                                             use_log=use_log)
            # GP prediction
            mu, var = gp_model.predict(y, X_test=X_test, return_var=True)
            std = np.sqrt(var)
            # renormalise outputs
            mu *= self.y_norm
            std *= self.y_norm

            # get SED light curve and mangle it
            sed_flux = np.interp(
                self.times_pred, sed_times, sed_lcs[band].values, left=0.0, right=0.0
            )
            lc_fit = sed_flux * mu
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
        # wavelength range goes a bit beyond the bluest and
        # reddest filters
        bands = self.filters.bands
        bluest_filter = self.filters[bands[0]]
        reddest_filter = self.filters[bands[-1]]
        dw = 20  # step in angstroms
        wavelengths_pred = np.arange(bluest_filter.wavelength.min() - 500,
                                     reddest_filter.wavelength.max() + 1000 + dw,
                                     dw
                                     )
        mangled_sed = {"flux": [], "flux_err": []}
        for phase in np.unique(self.sed.phase):
            # SED at the given phase
            phase_mask = self.sed.phase == phase
            sed_wave = self.sed.wave[phase_mask]
            sed_flux = self.sed.flux[phase_mask]

            # Mangling function - GP prediction from 'self.fit()'
            phases_pred = np.zeros_like(wavelengths_pred) + phase
            times_pred = phases_pred * (1 + self.z) + self.init_tmax
            X_test, y, _ = prepare_gp_inputs(times_pred, wavelengths_pred, 
                                             self.stacked_ratios, self.stacked_errors, 
                                             self.y_norm,
                                             use_log=self.use_log)
            # GP prediction
            mu, var = self.gp_model.predict(y, X_test=X_test, return_var=True)
            std = np.sqrt(var)
            # renormalise outputs
            mu *= self.y_norm
            std *= self.y_norm

            # convolute SED with mangling function
            mang_func = np.interp(sed_wave, wavelengths_pred, mu)
            mang_func_std = np.interp(sed_wave, wavelengths_pred, std)
            mangled_flux = mang_func * sed_flux
            mangled_flux_err = mang_func_std * sed_flux

            mangled_sed["flux"] += list(mangled_flux)
            mangled_sed["flux_err"] += list(mangled_flux_err)

        self.sed.flux = np.array(mangled_sed["flux"])
        self.sed.flux_err = np.array(mangled_sed["flux_err"])

    def _get_rest_lightcurves(self):
        """Calculates the rest-frame light curves after corrections."""
        self.sed.calculate_rest_lightcurves(self.filters)
        lcs_df_list = []
        fits_df_list = []
        for band in self.filters.bands:
            if band not in self.sed.rest_lcs:
                # the SED does not cover this band -> skip it
                continue
            mag_sys = self.filters[band].mag_sys
            zp = self.filters[band].calc_zp(mag_sys)

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

            fit = self.sed.rest_lcs_fit
            fit_df = pd.DataFrame(
                {
                    "time": fit.phase.values,
                    "flux": fit[band].values,
                    "flux_err": fit[f"{band}_err"].values,
                }
            )
            fit_df["zp"] = zp
            fit_df["band"] = band
            fit_df["mag_sys"] = mag_sys
            fits_df_list.append(fit_df)

        self.rest_lcs = Lightcurves(pd.concat(lcs_df_list))
        self.rest_lcs_fits = Lightcurves(pd.concat(fits_df_list))

    def _extract_lc_params(self):
        ########################
        # Estimate B-band Peak #
        ########################
        times_pred = self.rest_lcs.Bessell_B.times * (1 + self.z) + self.init_tmax
        rest_eff_wave = self.filters.Bessell_B.eff_wave * (1 + self.z)
        wavelengths_pred = np.zeros_like(times_pred) + rest_eff_wave
        # arrays for GP predictions
        X_test, y, _ = prepare_gp_inputs(times_pred, wavelengths_pred, 
                                         self.stacked_ratios, self.stacked_errors, 
                                         self.y_norm,
                                         use_log=self.use_log)
        # GP prediction
        mu, cov = self.gp_model.predict(y, X_test=X_test, return_cov=True)
        # renormalise outputs
        mu *= self.y_norm
        cov *= self.y_norm ** 2
        """
        # I don't think this is correct
        fluxes = self.rest_lcs.Bessell_B.fluxes
        flux_errors = self.rest_lcs.Bessell_B.flux_errors
        norm = flux_errors ** 2  / np.diag(cov)
        #cov *= norm



        # monte-carlo sampling
        mc_lcs = np.random.multivariate_normal(fluxes, cov, size=5000)
        tmax_list = []
        for lc in mc_lcs:
            peak_ids = peak.indexes(lc, thres=0.3, min_dist=len(times_pred) // 2)
            if len(peak_ids) == 0:
                # if no peak is found, just use the maximum
                max_id = np.argmax(lc)
            else:
                max_id = peak_ids[0]
            tmax_list.append(times_pred[max_id])
        # save time of maximum
        self.tmax = np.nanmean(tmax_list)
        self.tmax_err = np.nanstd(tmax_list)
        """
    
    def _extract_lc_params2(self):
        """Calculates the light-curves parameters.

        Estimation of B-band peak apparent magnitude, stretch, colour and colour-stretch parameters.
        An interpolation of the corrected light curves is done as well as part of this process.
        """
        self.rest_lcs.get_lc_params()
        band1 = "Bessell_B"
        band2 = "Bessell_V"
        self.tmax = self.init_tmax
        self.rest_lcs_fits[band1].get_max()
        delta_tmax = self.rest_lcs_fits[band1].tmax
        tmax_err = self.rest_lcs_fits[band1].tmax_err
        self.tmax_err = np.sqrt(delta_tmax**2 + tmax_err**2)

        self.mmax = self.rest_lcs[band1].mmax
        self.mmax_err = self.rest_lcs[band1].mmax_err
        self.dm15 = self.rest_lcs[band1].dm15
        self.dm15_err = self.rest_lcs[band1].dm15_err
        self.colour, self.colour_err = self.rest_lcs.get_max_colour(band1, band2)
        self.stretch, self.stretch_err = self.rest_lcs.get_colour_stretch(band1, band2)

        self.lc_parameters = {
            "tmax": np.round(self.tmax, 3),
            "tmax_err": np.round(self.tmax_err, 3),
            "Bmax": np.round(self.mmax, 3),
            "Bmax_err": np.round(self.mmax_err, 3),
            "dm15": np.round(self.dm15, 3),
            "dm15_err": np.round(self.dm15_err, 3),
            "colour": np.round(self.colour, 3),
            "colour_err": np.round(self.colour_err, 3),
            "sBV": np.round(self.stretch, 3),
            "sBV_err": np.round(self.stretch_err, 3),
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
        except:
            fits = self.init_lc_fits
        finally:
            if init_fits is True:
                fits = self.init_lc_fits

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
            fit_err_par = dict(alpha=0.5, color=colour)

            # light curves
            ax.errorbar(x, y, yerr, label=band, **data_par)
            ax.plot(x_fit, y_fit, **fit_par)
            ax.fill_between(x_fit, y_fit - yerr_fit, y_fit + yerr_fit, **fit_err_par)

            # residuals
            res = y - np.interp(x, x_fit, y_fit)
            ax2.errorbar(x, res, yerr, **data_par)
            ax2.plot(x_fit, np.zeros_like(y_fit), **fit_par)
            ax2.fill_between(x_fit, -yerr_fit, yerr_fit, **fit_err_par)

            for axis in [ax, ax2]:
                axis.axvline(
                    x=self.init_tmax - t_offset, color="k", linestyle="--", alpha=0.4
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
        columns = ["time", "flux", "flux_err", "mag", "mag_err", "zp", "band"]

        for band in self.bands:
            band_info = self.lc_fits[band]
            band_dict = {key: band_info[key] for key in columns}
            band_df = pd.DataFrame(band_dict)

            rounding_dict = {key: 3 for key in ["time", "mag", "mag_err", "zp"]}
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
        columns = ["time", "flux", "flux_err", "mag", "mag_err", "zp", "band"]

        for band in self.bands:
            band_info = self.rest_lcs[band]
            band_dict = {key: band_info[key] for key in columns}
            band_df = pd.DataFrame(band_dict)

            rounding_dict = {key: 3 for key in ["time", "mag", "mag_err", "zp"]}
            band_df = band_df.round(rounding_dict)
            df_list.append(band_df[columns])

        df_fits = pd.concat(df_list)
        df_fits.to_csv(output_file, sep="\t", index=False)
