import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import piscola
from .extinction_correction import redden, deredden, calculate_ebv
from .gaussian_process import gp_lc_fit


class SEDTemplate(object):
    """Spectral energy distribution (SED) class.

    This is used for correcting a supernova's multi-colout light curves.
    """

    def __init__(self, z=0.0, ra=None, dec=None, template="conley09f"):
        """
        Parameters
        ----------
        z: float, default ``0.0``
            Redshift.
        ra: float, default ``None``
            Right ascension.
        dec: float, default ``None``
            Declination.
        template: str, default ``conley09f``
            Name of the spectral energy distribution (SED) template.
            E.g., ``conley09f``, ``jla``, etc.
        """
        self.z = z
        self.ra = ra
        self.dec = dec

        self.set_sed_template(template)
        self.redshifted = False
        self.extincted = False

    def __repr__(self):
        return f"name: {self.name}, z: {self.z:.5}, ra: {self.ra}, dec: {self.dec}"

    def __getitem__(self, item):
        return getattr(self, item)

    def show_available_templates(self):
        """Prints all the available SED templates in the
        ``templates`` directory.
        """
        path = piscola.__path__[0]
        template_path = os.path.join(path, "templates")
        available_tamples = [
            name
            for name in os.listdir(template_path)
            if os.path.isdir(os.path.join(template_path, name))
        ]
        print("List of available SED templates:", available_tamples)

    def set_sed_template(self, template):
        """Sets the SED template to be used for the mangling function.

        Parameters
        ----------
        template : str
            Template name. E.g., ``conley09f``, ``jla``, etc.
        """
        # This can be modified to accept other templates
        pisco_path = piscola.__path__[0]
        sed_file = glob.glob(
            os.path.join(pisco_path, "templates", template, "sed_template.*")
        )[0]
        self.data = pd.read_csv(
            sed_file, delim_whitespace=True, names=["phase", "wave", "flux"]
        )
        self.phase = self.data.phase.values
        self.wave = self.data.wave.values
        self.flux = self.data.flux.values
        self.flux_err = np.zeros_like(self.flux)
        self.name = template

        readme_file = os.path.join(pisco_path, "templates", template, "README.txt")
        if os.path.isfile(readme_file):
            with open(readme_file, "r") as file:
                self.comments = file.read()
        else:
            self.comments = ""

    def redshift(self):
        """Redshifts the SED template if not already redshifted."""
        message = "The SED template is already redshifted."
        assert not self.redshifted, message

        self.phase *= 1 + self.z
        self.wave *= 1 + self.z
        self.flux /= 1 + self.z
        self.redshifted = True

    def deredshift(self):
        """De-redshifts the SED template if not already de-redshifted."""
        message = "The SED template is not redshifted."
        assert self.redshifted, message

        self.phase /= 1 + self.z
        self.wave /= 1 + self.z
        self.flux *= 1 + self.z
        self.redshifted = False

    def apply_extinction(
        self, scaling=0.86, reddening_law="fitzpatrick99", r_v=3.1, ebv=None
    ):
        """Applies Milky-Way extinction to the SED template if not already applied.

        Parameters
        ----------
        scaling: float, default ``0.86``
            Calibration of the Milky Way dust maps. Either ``0.86``
            for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
            dust map of Schlegel, Fikbeiner & Davis (1998).
        reddening_law: str, default ``fitzpatrick99``
            Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989),
            ``odonnell94`` (O’Donnell 1994), ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00``
            (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with :math:`R_V = 3.1`.)
        r_v : float, default ``3.1``
            Total-to-selective extinction ratio (:math:`R_V`)
        ebv : float, default ``None``
            Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.
        """

        message = "The SED template is already extincted."
        assert not self.extincted, message

        for phase in np.unique(self.phase):
            mask = self.phase == phase
            self.flux[mask] = redden(
                self.wave[mask],
                self.flux[mask],
                self.ra,
                self.dec,
                scaling,
                reddening_law,
                r_v=r_v,
                ebv=ebv,
            )
        if not ebv:
            self.ebv = calculate_ebv(self.ra, self.dec, scaling)
        else:
            self.ebv = ebv
        self.scaling = 0.86
        self.reddening_law = reddening_law
        self.r_v = r_v
        self.extinction_corrected = True

    def correct_extinction(self):
        """Corrects for Milky-Way extinction to the SED template if not already corrected.

        The same parameters used to apply extinction are used to correct it.
        """
        message = "The SED template is not extincted."
        assert self.extinction_corrected, message

        for phase in np.unique(self.phase):
            mask = self.phase == phase
            self.flux[mask] = deredden(
                self.wave[mask],
                self.flux[mask],
                self.ra,
                self.dec,
                self.scaling,
                self.reddening_law,
                r_v=self.r_v,
                ebv=self.ebv,
            )
        self.extinction_corrected = False

    def get_phase_data(self, phase, include_err=False):
        """Extracts the SED data for a given phase.

        The phase is given by the epochs of the SED template.

        Parameters
        ----------
        phase: int or float
            Phase of the SED.
        include_err: bool, default ``False``
            Whether or not to include uncertainties.

        Returns
        -------
        wave: array-like
            SED's wavelength range at the given phase.
        flux: array-like
            SED's flunx density at the given phase.
        flux_err: array-like
            Associated uncertainty in flux density. Only returned
            if ``include_err==True``.
        """
        mask = self.phase == phase
        wave, flux = self.wave[mask].copy(), self.flux[mask].copy()

        if not include_err:
            return wave, flux
        else:
            flux_err = self.flux_err[mask].copy()
            return wave, flux, flux_err

    def plot_sed(self, phase=0.0):
        """Plots the SED template at the given phase.

        The SED is shown at its current state, e.g. redshifted
        and/or extincted.

        Parameters
        ----------
        phase: int or float
            Phase of the SED.
        """
        err_message = f"Phase not found: {np.unique(self.phase)}"
        assert phase in self.phase, err_message

        wave, flux = self.get_phase_data(phase)
        plt.plot(wave, flux)
        plt.show()

    def calculate_obs_lightcurves(
        self, filters, scaling=0.86, reddening_law="fitzpatrick99", r_v=3.1, ebv=None
    ):
        """Calculates the multi-colour light curves of the SED as if
        it were observed by a telescope.

        Parameters
        ----------
        filters: list-like
            Filters used.
        scaling: float, default ``0.86``
            Calibration of the Milky Way dust maps. Either ``0.86``
            for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
            dust map of Schlegel, Fikbeiner & Davis (1998).
        reddening_law: str, default ``fitzpatrick99``
            Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989),
            ``odonnell94`` (O’Donnell 1994), ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00``
            (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with :math:`R_V` = 3.1.)
        r_v : float, default ``3.1``
            Total-to-selective extinction ratio (:math:`R_V`)
        ebv : float, default ``None``
            Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.

        Returns
        -------

        """
        if not self.redshifted:
            self.redshift()
        if not self.extincted:
            self.apply_extinction(scaling, reddening_law, r_v, ebv)

        # obs. light curves
        photometry = {band: [] for band in filters.bands}
        phases = np.unique(self.phase)
        for band in filters.bands:
            for phase in phases:
                wave, flux = self.get_phase_data(phase)
                obs_flux = filters[band].integrate_filter(wave, flux)
                photometry[band].append(obs_flux)

        photometry["phase"] = phases
        photometry_df = pd.DataFrame(photometry)
        self.obs_lcs = photometry_df

        # GP fit for interpolation
        fit_phot = {band: None for band in filters.bands}
        for band in filters.bands:
            flux = photometry_df[band].values
            # assuming no errors in the observations or in the fit
            phases_pred, flux_pred, _ = gp_lc_fit(phases, flux)
            fit_phot[band] = flux_pred

        fit_phot["phase"] = phases_pred
        fit_phot_df = pd.DataFrame(fit_phot)
        self.obs_lcs_fit = fit_phot_df

    def calculate_rest_lightcurves(self, filters):
        """Calculates rest-frame, corrected light curves of the SED.

        Parameters
        ----------
        filters: list-like
            Filters to use.
        """
        if self.extincted:
            self.correct_extinction()
        if self.redshifted:
            self.deredshift()

        # restframe light curves
        photometry = {band: [] for band in filters.bands}
        photometry.update({f"{band}_err": [] for band in filters.bands})
        phases = np.unique(self.phase)
        for band in filters.bands:
            filt = filters[band]
            for phase in phases:
                wave, flux, flux_err = self.get_phase_data(phase, include_err=True)
                rest_flux = filt.integrate_filter(wave, flux)
                rest_flux_err = filt.integrate_filter(wave, flux_err)
                photometry[band].append(rest_flux)
                photometry[f"{band}_err"].append(rest_flux_err)

        photometry["phase"] = phases
        photometry_df = pd.DataFrame(photometry)

        # GP fit for interpolation
        fit_phot = {band: None for band in filters.bands}
        for band in filters.bands:
            flux = photometry_df[band].values
            flux_err = photometry_df[f"{band}_err"].values
            # errors are underestimated as they are already correlated
            phases_pred, flux_pred, flux_pred_err = gp_lc_fit(phases, flux, flux_err)
            fit_phot[band] = flux_pred
            fit_phot[f"{band}_err"] = flux_pred_err

        fit_phot["phase"] = phases_pred
        fit_phot_df = pd.DataFrame(fit_phot)

        self.rest_lcs = photometry_df
        self.rest_lcs_fit = fit_phot_df
