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