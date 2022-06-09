import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import flux2mag

class lightcurve(object):
    """Light curve class.
    """
    def __init__(self, band, lc_file):
        self.band = band

        lc_df = pd.read_csv(lc_file, delim_whitespace=True,
                            skiprows=2)
        data = lc_df[lc_df.band == band]
        self.time = data.time.values
        self.flux = data.flux.values
        self.flux_err = data.flux_err.values
        self.zp = float(data.zp.unique()[0])
        self.mag, self.mag_err = flux2mag(self.flux, self.zp,
                                          self.flux_err)
        self.mag_sys = data.mag_sys.unique()[0]

    def __repr__(self):
        return f'band: {self.band}, zp: {self.zp}, mag_sys: {self.mag_sys}'

    def __getitem__(self, item):
        return getattr(self, item)

    def mask_lc(self, mask):
        self.masked_time = self.time.copy()[mask]
        self.masked_flux = self.flux.copy()[mask]
        self.masked_flux_err = self.flux_err.copy()[mask]
        self.masked_mag = self.mag.copy()[mask]
        self.masked_mag_err = self.mag_err.copy()[mask]


class lightcurves(object):
    """Multi-colour light curves class.
    """
    def __init__(self, lc_file):
        lc_df = pd.read_csv(lc_file, delim_whitespace=True,
                            skiprows=2)
        self.bands = lc_df.band.unique()

        for band in self.bands:
            lc = lightcurve(band, lc_file)
            setattr(self, band, lc)

    def __repr__(self):
        return str(self.bands)

    def __getitem__(self, item):
        return getattr(self, item)


class fitted_lightcurve(object):
    def __init__(self, band, time, flux, flux_err, zp):
        self.band = band
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.zp = zp
        mag, mag_err = flux2mag(flux, zp, flux_err)
        self.mag = mag
        self.mag_err = mag_err

    def __repr__(self):
        attrs = vars(self)
        rep = ', '.join("%s" % key for key in attrs.keys())
        return rep

    def __getitem__(self, item):
        return getattr(self, item)

class fitted_lightcurves(object):
    def __init__(self, fits_dict):
        self.bands = list(fits_dict.keys())

        for band, lc_dict in fits_dict.items():
            lc_fit = fitted_lightcurve(band, *lc_dict.values())
            setattr(self, band, lc_fit)

    def __repr__(self):
        return str(self.bands)

    def __getitem__(self, item):
        return getattr(self, item)