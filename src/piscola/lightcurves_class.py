import numpy as np
from peakutils import peak
import warnings

from .utils import flux2mag

class Lightcurve(object):
    """Light curve class.
    """
    def __init__(self, band, lcs_df):
        self.band = band

        data = lcs_df[lcs_df.band == band]
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

    def get_max(self):
        peak_ids = peak.indexes(-self.mag, thres=.3,
                               min_dist=len(self.time) // 3)
        if len(peak_ids)==0:
            self.mmax = self.mmax_err = np.nan
            self.tmax = np.nan
        else:
            self.mmax = self.mag[peak_ids[0]]
            self.mmax_err = self.mag_err[peak_ids[0]]
            self.tmax = self.time[peak_ids[0]]

    def get_dm15(self):
        self.get_max()
        if np.isnan(self.tmax):
            self.dm15 = self.dm15_err = np.nan
        else:
            phase = self.time - self.tmax
            if any(np.abs(phase-15) < 0.5):
                dm15_id = np.argmin(np.abs(phase-15))
                self.dm15 = self.mag[dm15_id]
                self.dm15_err = self.mag_err[dm15_id]
            else:
                self.dm15 = self.dm15_err = np.nan


class Lightcurves(object):
    """Multi-colour light curves class.
    """
    def __init__(self, lcs_df):
        self.bands = lcs_df.band.unique()

        for band in self.bands:
            lc = Lightcurve(band, lcs_df)
            setattr(self, band, lc)

    def __repr__(self):
        return str(self.bands)

    def __getitem__(self, item):
        return getattr(self, item)

    def get_max_colour(self, band1, band2):
        cond1 = band1 in self.bands
        cond2 = band2 in self.bands
        assert cond1 and cond2, f"band(s) not in {self.bands}"

        self[band1].get_max()
        tmax = self[band1].tmax
        mmax = self[band1].mmax
        mmax_err = self[band1].mmax_err
        if np.isnan(tmax):
            warnings.warn(f"cannot estimate time of peak for {band1}")
            colour = colour_err = np.nan
        else:
            rel_phase = self[band2].time - tmax
            if any(np.abs(rel_phase) < 0.5):
                band2_id = np.argmin(np.abs(rel_phase))
                mag2 = self[band2].mag[band2_id]
                mag2_err = self[band2].mag_err[band2_id]
                colour = mmax - mag2
                colour_err = np.sqrt(mmax_err**2 + mag2_err**2)
            else:
                warnings.warn(f"{band2} does not have data at time of peak for {band1}")
                colour = colour_err = np.nan

        return colour, colour_err

    def get_colour_stretch(self, band1, band2):
        cond1 = band1 in self.bands
        cond2 = band2 in self.bands
        assert cond1 and cond2, f"band(s) not in {self.bands}"

        mag1 = self[band1].mag
        mag1_err = self[band1].mag_err
        mag2 = self[band2].mag
        mag2_err = self[band2].mag_err
        colour_curve = mag1 - mag2
        colour_curve_err = mag2 = np.sqrt(mag1_err**2 + mag2_err**2)

        self[band1].get_max()
        tmax = self[band1].tmax
        time = self[band1].time

        if np.isnan(tmax):
            warnings.warn(f"cannot estimate time of peak for {band1}")
            stretch = stretch_err = np.nan
        else:
            mask = time > tmax
            colour_curve = np.copy(colour_curve[mask])
            time = np.copy(time[mask])
            peak_ids = peak.indexes(colour_curve, thres=.3,
                                    min_dist=len(time) // 3)
            if len(peak_ids) > 0:
                colour_tmax = time[peak_ids[0]]
                stretch = (colour_tmax - tmax)/30
                stretch_err = 0.0
            else:
                warnings.warn(f"The peak in the colour curve was not found")
                stretch = stretch_err = np.nan

        return stretch, stretch_err


    def get_lc_params(self):

        for band in self.bands:
            self[band].get_max()
            self[band].get_dm15()