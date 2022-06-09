import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import piscola
from .extinction_correction import redden, deredden, calculate_ebv
from .gaussian_process import gp_lc_fit

class sed_template(object):
    """Spectral energy distribution (SED) class
    for correcting a supernova.

    Parameters
    ----------
    template : str, default ``conley09f``
        Template name. E.g., ``conley09f``, ``jla``, etc.
    """

    def __init__(self, z=0.0, ra=None, dec=None, template='conley09f'):
        self.z = z
        self.ra = ra
        self.dec = dec

        self.set_sed_template(template)
        self.redshifted = False
        self.extinction_corrected = False

    def __repr__(self):
        return f'name: {self.name}, z: {self.z:.5}, ra: {self.ra}, dec: {self.dec}'

    def __getitem__(self, item):
        return getattr(self, item)

    def show_available_templates(self):
        """Prints all the available SED templates in the
        ``templates`` directory.
        """
        path = piscola.__path__[0]
        template_path = os.path.join(path, "templates")
        available_tamples = [name for name in os.listdir(template_path)
                             if os.path.isdir(os.path.join(template_path, name))]
        print('List of available SED templates:', available_tamples)

    def set_sed_template(self, template):
        """Sets the SED template to be used for the mangling function.

        Parameters
        ----------
        template : str
            Template name. E.g., ``conley09f``, ``jla``, etc.
        """
        # This can be modified to accept other templates
        pisco_path = piscola.__path__[0]
        sed_file = glob.glob(os.path.join(pisco_path, 'templates',
                                          template, 'snflux_1a.*'))[0]
        self.data = pd.read_csv(sed_file, delim_whitespace=True,
                                names=['phase', 'wave', 'flux'])
        self.phase = self.data.phase.values
        self.wave = self.data.wave.values
        self.flux = self.data.flux.values
        self.name = template

        readme_file = os.path.join(pisco_path, 'templates',
                                   template, 'README.txt')
        if os.path.isfile(readme_file):
            with open(readme_file, 'r') as file:
                self.comments = file.read()
        else:
            self.comments = ''

    def redshift(self):

        message = 'The SED template is already redshifted.'
        assert not self.redshifted, message

        self.phase *= (1 + self.z)
        self.wave *= (1 + self.z)
        self.flux /= (1 + self.z)
        self.redshifted = True

    def deredshift(self):

        message = 'The SED template is not redshifted.'
        assert self.redshifted, message

        self.phase /= (1 + self.z)
        self.wave /= (1 + self.z)
        self.flux *= (1 + self.z)
        self.redshifted = False

    def apply_extinction(self, scaling=0.86,
                         reddening_law='fitzpatrick99',
                         r_v=3.1, ebv=None):

        message = 'The SED template is already extincted.'
        assert not self.extinction_corrected, message

        for phase in np.unique(self.phase):
            mask = self.phase == phase
            self.flux[mask] = redden(self.wave[mask],
                                     self.flux[mask],
                                     self.ra, self.dec,
                                     scaling, reddening_law,
                                     r_v=r_v, ebv=ebv)
        if not ebv:
            self.ebv = calculate_ebv(self.ra, self.dec,
                                     scaling)
        else:
            self.ebv = ebv
        self.scaling = 0.86
        self.reddening_law = reddening_law
        self.r_v = r_v
        self.extinction_corrected = True

    def correct_extinction(self):

        message = 'The SED template is not extincted.'
        assert self.extinction_corrected, message

        for phase in np.unique(self.phase):
            mask = self.phase == phase
            self.flux[mask] = deredden(self.wave[mask],
                                       self.flux[mask],
                                       self.ra, self.dec,
                                       self.scaling,
                                       self.reddening_law,
                                       r_v=self.r_v, ebv=self.ebv)
        self.extinction_corrected = False

    def get_phase_data(self, phase):

        mask = self.phase == phase
        wave, flux = self.wave[mask].copy(), self.flux[mask].copy()

        return wave, flux

    def plot_sed(self, phase=0.0):

        err_message = f'Phase not found: {np.unique(self.phase)}'
        assert phase in self.phase, err_message

        wave, flux = self.get_phase_data(phase)
        plt.plot(wave, flux)
        plt.show()

    def calculate_obs_lightcurves(self, filters, scaling=0.86,
                                 reddening_law='fitzpatrick99',
                                 r_v=3.1, ebv=None):

        self.redshift()
        self.apply_extinction(scaling, reddening_law, r_v, ebv)

        # obs. light curves
        photometry = {band:[] for band in filters.bands}
        phases = np.unique(self.phase)
        for band in filters.bands:
            for phase in phases:
                wave, flux = self.get_phase_data(phase)
                obs_flux = filters[band].integrate_filter(wave, flux)
                photometry[band].append(obs_flux)

        photometry['phase'] = phases
        photometry_df = pd.DataFrame(photometry)

        # GP fit for interpolation
        fit_phot = {band:None for band in filters.bands}
        for band in filters.bands:
            flux = photometry_df[band].values
            # assuming no errors in the observations or in the fit
            phases_pred, flux_pred, _ = gp_lc_fit(phases, flux)
            fit_phot[band] = flux_pred

        fit_phot['phase'] = phases_pred
        fit_phot_df = pd.DataFrame(fit_phot)

        self.obs_lcs = photometry_df
        self.obs_lcs_fit = fit_phot_df
        self.correct_extinction()
        self.deredshift()
