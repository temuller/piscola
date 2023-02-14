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
from .gaussian_process import gp_2d_fit


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
            self.filters = MultiFilters(self.lcs.bands)

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
        self.sed.calculate_obs_lightcurves(self.filters)

    def __repr__(self):
        rep = (
            f"name: {self.name}, z: {self.z:.5}, "
            f"ra: {self.ra:.5}, dec: {self.dec:.5}\n"
        )
        return rep

    def __getitem__(self, item):
        return getattr(self, item)

    def save_sn(self, path=None):
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

        self._init_fits = None
        self._fit_results = None

        outfile = os.path.join(path, f"{self.name}.pisco")
        with bz2.BZ2File(outfile, "wb") as pfile:
            if sys.version_info.minor < 8:
                protocol = 4
            else:
                protocol = pickle.HIGHEST_PROTOCOL
            pickle.dump(self, pfile, protocol)

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

            self.lcs[band].flux = change_zp(self.lcs[band]["flux"], current_zp, new_zp)
            self.lcs[band].flux_err = change_zp(
                self.lcs[band]["flux_err"], current_zp, new_zp
            )
            self.lcs[band].zp = new_zp

    def _stack_lcs(self):
        """Stacks of light-curve properties

        Times, wavelengths, fluxes, magnitudes and errors are
        stacked for 2D fitting.
        """
        time = np.hstack([self.lcs[band].time for band in self.bands])
        wave = np.hstack(
            [
                [self.filters[band].eff_wave] * len(self.lcs[band].time)
                for band in self.bands
            ]
        )
        flux = np.hstack([self.lcs[band].flux for band in self.bands])
        flux_err = np.hstack([self.lcs[band].flux_err for band in self.bands])
        mag = np.hstack([self.lcs[band].mag for band in self.bands])
        mag_err = np.hstack([self.lcs[band].mag_err for band in self.bands])

        self._stacked_time = time
        self._stacked_wave = wave
        self._stacked_flux = flux
        self._stacked_flux_err = flux_err
        self._stacked_mag = mag
        self._stacked_mag_err = mag_err

    def _fit_lcs(self, kernel1="matern52", kernel2="squaredexp", gp_mean="mean"):
        """Fits the multi-colour light-curve data with gaussian process.

        The time of rest-frame B-band peak luminosity is estimated by finding where the derivative is equal to zero.

        Parameters
        ----------
        kernel : str, default ``matern52``
            Kernel to be used in the **time**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        kernel2 : str, default ``matern52``
            Kernel to be used in the **wavelengt**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        gp_mean : str, default ``mean``
            Gaussian process mean function. Either ``mean``, ``max`` or ``min``.
        """
        self._stack_lcs()
        self._x2_ext = (1000, 2000)
        for i in range(5):
            self._x1_ext = (5*(i+1), 10)
            timeXwave, lc_mean, lc_std, gp = gp_2d_fit(
                self._stacked_time,
                self._stacked_wave,
                self._stacked_flux,
                self._stacked_flux_err,
                kernel1,
                kernel2,
                gp_mean,
                self._x1_ext,
                self._x2_ext,
            )
            self.init_gp = gp
            self._init_fits = {}
            self._init_fits["timeXwave"] = timeXwave
            self._init_fits["lc_mean"] = lc_mean
            self._init_fits["lc_std"] = lc_std

            # Estimate B-band Peak
            sed_wave, sed_flux = self.sed.get_phase_data(0.0)
            B_eff_wave = self.filters.Bessell_B.calc_eff_wave(sed_wave, sed_flux)

            times, waves = timeXwave.T[0], timeXwave.T[1]
            wave_ind = np.argmin(np.abs(B_eff_wave * (1 + self.z) - waves))
            eff_wave = waves[wave_ind]
            Bmask = waves == eff_wave

            Btime, Bflux, Bflux_err = times[Bmask], lc_mean[Bmask], lc_std[Bmask]
            peak_ids = peak.indexes(Bflux, thres=0.3, min_dist=len(Btime) // 2)

            if len(peak_ids) != 0:
                peak_id = peak_ids[0]
                break

        if len(peak_ids) == 0:
            # if still no peak is found, use the maximum
            peak_id = np.argmax(Bflux)
        self.init_tmax = np.round(Btime[peak_id], 3)

        # get tmax_err
        Bflux = Bflux[peak_id]
        # use only data around peak
        mask = (Btime > self.init_tmax - 5) & (Btime < self.init_tmax + 5)
        Btime = Btime[mask]
        brightest_flux = (Bflux + Bflux_err)[mask]
        id_err = np.argmin(np.abs(brightest_flux - Bflux))
        self.tmax_err = np.abs(Btime[id_err] - self.init_tmax)

        # inital light-curve fits
        times, waves = timeXwave.T
        fits_df_list = []
        for band in self.bands:
            wave_ind = np.argmin(np.abs(self.filters[band]["eff_wave"] - waves))
            eff_wave = waves[wave_ind]
            mask = waves == eff_wave
            time, mean, std = times[mask], lc_mean[mask], lc_std[mask]

            fit_df = pd.DataFrame({"time": time, "flux": mean, "flux_err": std})
            fit_df["zp"] = self.lcs[band].zp
            fit_df["band"] = band
            fit_df["mag_sys"] = self.lcs[band].mag_sys
            fits_df_list.append(fit_df)

        self._init_lc_fits = Lightcurves(pd.concat(fits_df_list))


    def fit(self, kernel1="matern52", kernel2="squaredexp", gp_mean="mean"):
        """Fits and corrects the multi-colour light curves.

        The corrections include Milky-Way dust extinction and mangling of the SED.
        Rest-frame light curves and parameters are calculated as end products.

        Parameters
        ----------
        kernel : str, default ``matern52``
            Kernel to be used in the **time**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        kernel2 : str, default ``matern52``
            Kernel to be used in the **wavelengt**-axis when fitting the light curves with gaussian process. E.g.,
            ``matern52``, ``matern32``, ``squaredexp``.
        gp_mean : str, default ``mean``
            Gaussian process mean function. Either ``mean``, ``max`` or ``min``.
        """
        self._fit_lcs(kernel1, kernel2, gp_mean)  # to get initial tmax

        sed_lcs = self.sed.obs_lcs_fit  # interpolated light curves
        sed_times = sed_lcs.phase.values + self.init_tmax

        times, waves = [], []
        flux_ratios, flux_err = [], []
        for band in self.bands:
            # this mask avoids observations beyond the SED template limits
            mask = self.lcs[band].time <= sed_times.max()
            self.lcs[band].mask_lc(mask, copy=True)
            sed_flux = np.interp(
                self.lcs[band].masked_time,
                sed_times,
                sed_lcs[band].values,
                left=0.0,
                right=0.0,
            )

            times.append(self.lcs[band].masked_time)
            waves.append(
                [self.filters[band].eff_wave] * len(self.lcs[band].masked_time)
            )
            flux_ratios.append(self.lcs[band].masked_flux / sed_flux)
            flux_err.append(self.lcs[band].masked_flux_err / sed_flux)

        # prepare data for 2D fit
        self._stacked_times = np.hstack(times)
        self._stacked_waves = np.hstack(waves)
        self._stacked_ratios = np.hstack(flux_ratios)
        self._stacked_err = np.hstack(flux_err)

        timeXwave, mf_mean, mf_std, gp = gp_2d_fit(
            self._stacked_times,
            self._stacked_waves,
            self._stacked_ratios,
            self._stacked_err,
            kernel1,
            kernel2,
            gp_mean,
            self._x1_ext,
            self._x2_ext,
        )
        self.gp = gp
        self._fit_results = {}
        self._fit_results["timeXwave"] = timeXwave
        self._fit_results["mf_mean"] = mf_mean
        self._fit_results["mf_std"] = mf_std

        times, waves = timeXwave.T
        fits_df_list = []
        for band in self.bands:
            wave_ind = np.argmin(np.abs(self.filters[band]["eff_wave"] - waves))
            eff_wave = waves[wave_ind]
            mask = waves == eff_wave
            time, mf, std = times[mask], mf_mean[mask], mf_std[mask]

            sed_flux = np.interp(
                time, sed_times, sed_lcs[band].values, left=0.0, right=0.0
            )
            lc_fit = sed_flux * mf
            lc_std = sed_flux * std

            fit_df = pd.DataFrame({"time": time, "flux": lc_fit, "flux_err": lc_std})
            fit_df["zp"] = self.lcs[band].zp
            fit_df["band"] = band
            fit_df["mag_sys"] = self.lcs[band].mag_sys
            fits_df_list.append(fit_df)

        self.lc_fits = Lightcurves(pd.concat(fits_df_list))
        self._mangle_sed()
        self._get_rest_lightcurves()
        self._extract_lc_params()


    def _calc_fits_results(self, lc):

        if lc:
            y = self._stacked_flux
            x1, x2 = self._stacked_time, self._stacked_wave
            gp = self.init_gp
        else:
            y = self._stacked_ratios
            x1, x2 = self._stacked_times, self._stacked_waves
            gp = self.gp

        y_norm = y.max()
        y /=  y_norm

        x1_min, x1_max = x1.min() - self._x1_ext[0], x1.max() + self._x1_ext[1]
        x2_min, x2_max = x2.min() - self._x2_ext[0], x2.max() + self._x2_ext[1]

        # x-axis prediction array
        step1 = 0.1  # in days
        step2 = 10  # in angstroms
        x1_pred = np.arange(x1_min, x1_max + step1, step1)
        x2_pred = np.arange(x2_min, x2_max + step2, step2)
        X_predict = np.array(np.meshgrid(x1_pred, x2_pred))
        X_predict = X_predict.reshape(2, -1).T

        mean, std = gp.predict(y, X_predict, return_var=True)
        y_pred = mean * y_norm
        yerr_pred = std * y_norm

        if lc:
            results = {"timeXwave": X_predict,
                       "lc_mean": y_pred,
                       "lc_std": yerr_pred}
        else:
            results = {"timeXwave": X_predict,
                       "mf_mean": y_pred,
                       "mf_std": yerr_pred}

        return results

    @property
    def fit_results(self):
        if self._fit_results is None:
            self._fit_results = self._calc_fits_results(lc=False)

        return self._fit_results

    @property
    def init_fits(self):
        if self._init_fits is None:
            self._init_fits = self._calc_fits_results(lc=True)

        return self._init_fits

    def _mangle_sed(self):
        """Mangles the supernova SED."""
        times, waves = self._fit_results["timeXwave"].T
        mf_mean = self._fit_results["mf_mean"]
        mf_std = self._fit_results["mf_std"]

        # mangle SED in observer frame
        mangled_sed = {"flux": [], "flux_err": []}
        for phase in np.unique(self.sed.phase):
            # SED
            phase_mask = self.sed.phase == phase
            sed_wave = self.sed.wave[phase_mask]
            sed_flux = self.sed.flux[phase_mask]

            # mangling function
            phases = times - self.init_tmax
            phase_id = np.argmin(np.abs(phases - phase))
            phase_mask = phases == phases[phase_id]
            mang_wave = waves[phase_mask]
            mang_func = mf_mean[phase_mask]
            mang_func_std = mf_std[phase_mask]

            mang_func = np.interp(sed_wave, mang_wave, mang_func)
            mang_func_std = np.interp(sed_wave, mang_wave, mang_func_std)
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
            if "Bessell" in band:
                mag_sys = "BD17"
            else:
                mag_sys = self.lcs[band].mag_sys
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
        colours = palette1 + palette2 + palette3

        # shift in time for visualization purposes
        min_time = self.lcs[self.bands[1]].time.min()
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

            x = self.lcs[band].time - t_offset
            if not plot_mag:
                y_norm = change_zp(1.0, self.lcs[band]["zp"], ZP)
                y = self.lcs[band].flux * y_norm
                yerr = self.lcs[band].flux_err * y_norm
            else:
                y = self.lcs[band].mag
                yerr = self.lcs[band].mag_err
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

    def plot_fits(self, plot_mag=False, fig_name=None):
        """Plots the light-curves fits results.

        Plots the observed data for each band together with the gaussian process fits.
        The time of :math:`B`-band peak is shown as a vertical dashed line.

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
        colours = palette1 + palette2 + palette3

        # shift in time for visualization purposes
        tmax_str = str(self.init_tmax.astype(int))
        zeros = "0" * len(tmax_str[2:])
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
                y_norm = change_zp(1.0, self.lcs[band]["zp"], ZP)
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
