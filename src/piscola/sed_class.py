import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import piscola
from .extinction_correction import redden, deredden, calculate_ebv
from .gaussian_process import fit_single_lightcurve

def show_available_templates():
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

class SEDTemplate(object):
    """Spectral energy distribution (SED) class.

    This is used for correcting a supernova's multi-colout light curves.
    """

    def __init__(self, z=0.0, ra=None, dec=None, template="csp"):
        """
        Parameters
        ----------
        z: float, default ``0.0``
            Redshift.
        ra: float, default ``None``
            Right ascension.
        dec: float, default ``None``
            Declination.
        template: str, default ``csp``
            Name of the spectral energy distribution (SED) template.
            E.g., ``csp``, ``hsiao``, ``salt2``, ``salt3``, etc.
        """
        self.z = z
        self.ra = ra
        self.dec = dec

        self._set_sed_template(template)
        self.redshifted = False
        self.extincted = False
        self.mangled = False

    def __repr__(self):
        return f"name: {self.name}, z: {self.z:.5}, ra: {self.ra}, dec: {self.dec}"

    def __getitem__(self, item):
        return getattr(self, item)
    
    def _retrieve_template(self, template):
        """Helper function for retrieving the SED template.

        Useful for loading the initial SED again to save space when
        saving an SN object.

        Parameters
        ----------
        template : str
            Template name. E.g., ``csp``, ``hsiao``, ``salt2``, ``salt3``, etc.
        """
        pisco_path = piscola.__path__[0]
        sed_file = glob.glob(
            os.path.join(pisco_path, "templates", template, "sed_template.*")
        )[0]
        self.data = pd.read_csv(
            sed_file, sep='\s+', names=["phase", "wave", "flux"]
        )

        readme_file = os.path.join(pisco_path, "templates", template, "README.txt")
        if os.path.isfile(readme_file):
            with open(readme_file, "r") as file:
                self.comments = file.read()
        else:
            self.comments = ""

    def _set_sed_template(self, template):
        """Sets the SED template to be used for the mangling function.

        Parameters
        ----------
        template : str
            Template name. E.g., ``conley09f``, ``jla``, etc.
        """
        self._retrieve_template(template)

        self.phase = self.data.phase.values
        self.wave = self.data.wave.values
        self.flux = self.data.flux.values
        self.flux_err = np.zeros_like(self.flux)
        self.name = template

    def mask_sed(self, min_phase=None, max_phase=None, min_wave=None, max_wave=None):
        """Masks the SED phase and wavelength coverage.

        Both the minimum and maximum in time or wavenlength need
        to be given to mask the SED.

        Parameters
        ----------
        min_phase : float, default ``None``
            Minimum phase to include.
        max_phase : float, default ``None``
            Maximum phase to include.
        min_wave : float, default ``None``
            Minimum wavelength to include.
        max_wave : float, default ``None``
            Maximum wavelength to include.
        """
        if min_phase is not None and max_phase is not None:
            # mask phase
            phase_mask = (self.phase >= min_phase) & (self.phase <= max_phase)
            self.phase = self.phase[phase_mask]
            self.wave = self.wave[phase_mask]
            self.flux = self.flux[phase_mask]
            self.flux_err = self.flux_err[phase_mask]

        if min_wave is not None and max_wave is not None:
            # mask wavelength
            wave_mask = (self.wave >= min_wave) & (self.wave <= max_wave)
            self.phase = self.phase[wave_mask]
            self.wave = self.wave[wave_mask]
            self.flux = self.flux[wave_mask]
            self.flux_err = self.flux_err[wave_mask]

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
        assert self.extincted is False, message

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
        self.extincted = True

    def correct_extinction(self):
        """Corrects for Milky-Way extinction to the SED template if not already corrected.

        The same parameters used to apply extinction are used to correct it.
        """
        message = "The SED template is not extincted."
        assert self.extincted is True, message

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
        self.extincted = False

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
        self, filters, bands=None, scaling=0.86, reddening_law="fitzpatrick99", r_v=3.1, ebv=None
    ):
        """Calculates the multi-colour light curves of the SED as if
        it were observed by a telescope.

        Parameters
        ----------
        filters: ~piscola.filters_class.MultiFilters
            Filters for integrating flux.
        bands: list-like
            Bands to use.
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
        """
        # apply time dilation and extinction
        if self.redshifted is False:
            self.redshift()
        if self.extincted is False:
            self.apply_extinction(scaling, reddening_law, r_v, ebv)

        if bands is None:
            bands = filters.bands

        # get observed phases and prediction array
        photometry = {band: [] for band in bands}
        phases = np.unique(self.phase)
        dt = 0.1 * (1 + self.z)  # 0.1 days in rest-frame, moved to observer frame
        phases_pred = np.arange(phases.min(), phases.max() + dt, dt)
        for band in bands:
            obs_fluxes = []
            for phase in phases:
                # calculate photometry
                wave, flux = self.get_phase_data(phase)
                obs_fluxes.append(filters[band].integrate_filter(wave, flux))
            obs_fluxes = np.array(obs_fluxes)

            # GP fit for interpolating the light curves
            # assuming no errors in the observations or the fit
            gp_model = fit_single_lightcurve(phases, obs_fluxes, np.array([0.0]))
            y = obs_fluxes.copy() / obs_fluxes.max()
            mu = gp_model.predict(y, X_test=phases_pred)
            photometry[band] = mu * obs_fluxes.max()  # renormalise output

        # store light curves
        photometry["phase"] = phases_pred
        photometry_df = pd.DataFrame(photometry)
        self.obs_lcs = photometry_df

    def calculate_rest_lightcurves(self, filters):
        """Calculates rest-frame, corrected light curves of the SED.

        Parameters
        ----------
        filters: list-like
            Filters to use.
        """
        # correct for extinction and time dilation
        if self.extincted is True:
            self.correct_extinction()
        if self.redshifted is True:
            self.deredshift()

        # get rest-frame phases and prediction array
        photometry = {band: [] for band in filters.bands}
        photometry.update({f"{band}_err": [] for band in filters.bands})
        phases = np.unique(self.phase)
        dt = 0.1  # 0.1 days in rest-frame
        phases_pred = np.arange(phases.min(), phases.max() + dt, dt)
        for band in filters.bands:
            rest_fluxes, rest_flux_errors = [], []
            for phase in phases:
                # calculate photometry
                wave, flux, flux_err = self.get_phase_data(phase, include_err=True)
                try:
                    rest_fluxes.append(filters[band].integrate_filter(wave, flux))
                except:
                    # The SED has no coverage for this filter
                    continue
                rest_flux_errors.append(filters[band].integrate_filter(wave, flux_err))

            # remove empty bands
            if len(rest_fluxes) == 0:
                photometry.pop(band)
                photometry.pop(f"{band}_err")
                continue

            rest_fluxes = np.array(rest_fluxes)
            rest_flux_errors = np.array(rest_flux_errors)
            """
            # GP fit for interpolating the light curves
            # assuming no errors in the observations or the fit
            gp_model = fit_single_lightcurve(phases, rest_fluxes, rest_flux_errors)
            y = rest_fluxes.copy() / rest_fluxes.max()
            mu, var = gp_model.predict(y, X_test=phases_pred, return_var=True)
            photometry[band] = mu * rest_fluxes.max()  # renormalise output
            photometry[f"{band}_err"] = np.sqrt(var) * rest_fluxes.max()
            """
            # GP fit for interpolating the light curves
            # assuming no errors in the observations or the fit
            gp_model = fit_single_lightcurve(phases, rest_fluxes, np.zeros_like(rest_fluxes))
            y = rest_fluxes.copy() / rest_fluxes.max()
            mu = gp_model.predict(y, X_test=phases_pred)
            photometry[band] = mu * rest_fluxes.max()  # renormalise output
            # interpolate errors using S/N for smoother interpolation
            #signal_to_noise = np.interp(phases_pred, phases, (rest_fluxes + rest_fluxes.min())/rest_flux_errors)
            #photometry[f"{band}_err"] = np.abs((photometry[band] + rest_fluxes.min()) / signal_to_noise)
            photometry[f"{band}_err"] = np.interp(phases_pred, phases, rest_flux_errors)

        # store light curves
        photometry["phase"] = phases_pred
        photometry_df = pd.DataFrame(photometry)
        self.rest_lcs = photometry_df