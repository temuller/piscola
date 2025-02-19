import numpy as np
from peakutils import peak
import warnings

from .utils import flux_to_mag


class Lightcurve(object):
    """Light curve class."""

    def __init__(self, band, lcs_df):
        """

        Parameters
        ----------
        band: str
            Light-curve filter/band name.
        lcs_df: Dataframe
            Light-curve data: `times`, `fluxes`, `flux_errors`, `magnitudes`, `mag_errors`
            and `magnitude system`.
        """
        self.band = band

        data = lcs_df[lcs_df.band == band]
        self.times = data.time.values
        self.fluxes = data.flux.values
        self.flux_errors = data.flux_err.values
        self.zp = float(data.zp.unique()[0])
        self.magnitudes, self.mag_errors = flux_to_mag(self.fluxes, self.flux_errors, self.zp)
        self.mag_sys = data.mag_sys.unique()[0]

    def __repr__(self):
        return (f"band: {self.band}, zp: {self.zp:.5}, mag_sys: {self.mag_sys}"
                "\nother attributes: times, fluxes, flux_errors, magnitudes, mag_errors")

    def __getitem__(self, item):
        return getattr(self, item)

    def mask_lc(self, mask, copy=False):
        """Masks the light-curve data with the given mask.

        Parameters
        ----------
        mask: bool list
            Mask to apply to the light curve.
        mask: bool, default ``False``
            If ``True``, the masked light curves are saved on
            separate arrays.
        """
        if not copy:
            self.times = self.times[mask]
            self.fluxes = self.fluxes[mask]
            self.flux_errors = self.flux_errors[mask]
            self.magnitudes = self.magnitudes[mask]
            self.mag_errors = self.mag_errors[mask]
        else:
            self.masked_times = self.times[mask]
            self.masked_fluxes = self.fluxes[mask]
            self.masked_flux_errors = self.flux_errors[mask]
            self.masked_magnitudes = self.magnitudes[mask]
            self.masked_mag_errors = self.mag_errors[mask]

    def get_max(self):
        """Calculates the peak magnitude (:math:`m_{max}`) and
        its epoch (:math:`t_{max}`).
        """
        magnitudes = np.nan_to_num(self.magnitudes, nan=np.nanmean(self.magnitudes))
        peak_ids = peak.indexes(-magnitudes, thres=0.3, min_dist=len(self.times) // 3)
        if len(peak_ids) == 0:
            self.mmax = self.mmax_err = np.nan
            self.tmax = np.nan
            self.tmax_err = np.nan
        else:
            self.mmax = self.magnitudes[peak_ids[0]]
            self.mmax_err = self.mag_errors[peak_ids[0]]
            self.tmax = self.times[peak_ids[0]]

            # get tmax_err
            # use only data around peak
            mask = (self.times > self.tmax - 5) & (self.times < self.tmax + 5)
            times = self.times[mask]
            brightest_mag = (self.magnitudes - self.mag_errors)[mask]
            id_err = np.argmin(np.abs(brightest_mag - self.mmax))
            self.tmax_err = np.abs(times[id_err] - self.tmax)

    def get_dm15(self):
        r"""Calculates the classic parameter :math:`\Delta m_{15}`
        (Phillips 1993), but using the epoch of peak magnitude of
        the current light curve as reference (not necessarily :math:`B`
        band).
        """
        self.get_max()
        if np.isnan(self.tmax):
            self.dm15 = self.dm15_err = np.nan
        else:
            phases = self.times - self.tmax
            if any(np.abs(phases - 15) < 0.5):
                dm15_id = np.argmin(np.abs(phases - 15))
                self.dm15 = self.magnitudes[dm15_id] - self.mmax
                self.dm15_err = np.sqrt(self.mag_errors[dm15_id] ** 2 + self.mmax_err**2)
            else:
                self.dm15 = self.dm15_err = np.nan


class Lightcurves(object):
    """Multi-colour light curves class."""

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
        """Calculates the colour at the epoch of peak magnitude
        in ``band1``.

        The colour is :math:`(band1 - band2)` at the time of
        peak magnitude in ``band1``.

        Parameters
        ----------
        band1: str
            Name of first band.
        band2: str
            Name of second band.

        Returns
        -------
        colour: float
            Colour at the epoch of peak magnitude in ``band1``.
        colour_err: float
            Uncertainty on the colour parameter.
        """
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
            rel_phases = self[band2].times - tmax
            if any(np.abs(rel_phases) < 0.5):
                band2_id = np.argmin(np.abs(rel_phases))
                mag2 = self[band2].magnitudes[band2_id]
                mag2_err = self[band2].mag_errors[band2_id]
                colour = mmax - mag2
                colour_err = np.sqrt(mmax_err**2 + mag2_err**2)
            else:
                warnings.warn(f"{band2} does not have data at time of peak for {band1}")
                colour = colour_err = np.nan

        return colour, colour_err

    def get_colour_stretch(self, band1, band2):
        """Calculates the colour stretch parameter (Burns et al. 2014).

        The formula used is :math:`(T(band1-band2)_{max} - T_{max}/30)`, where
        :math:`T(band1−band2)_{max}` is the time of maximum (reddest colour)
        in the :math:`(band1−band2)` colour curve and :math:`T_{max}` is the
        epoch of peak magnitude in ``band1``.

        Parameters
        ----------
        band1: str
            Name of first band.
        band2: str
            Name of second band.

        Returns
        -------
        stretch: float
            Colour-stretch parameter.
        stretch_err: float
            Uncertainty on the colour-stretch parameter.
        """
        cond1 = band1 in self.bands
        cond2 = band2 in self.bands
        assert cond1 and cond2, f"band(s) not in {self.bands}"

        colour_curve = self[band1].magnitudes - self[band2].magnitudes
        colour_curve_err = np.sqrt(self[band1].mag_errors ** 2 + self[band2].mag_errors ** 2)

        self[band1].get_max()
        tmax = self[band1].tmax
        tmax_err = self[band1].tmax_err
        times = self[band1].times

        if np.isnan(tmax):
            warnings.warn(f"cannot estimate time of peak for {band1}")
            stretch = stretch_err = np.nan
        else:
            mask = times > tmax
            colour_curve = np.copy(colour_curve[mask])
            times = np.copy(times[mask])
            peak_ids = peak.indexes(colour_curve, thres=0.3, min_dist=len(times) // 3)
            if len(peak_ids) > 0:
                colour_tmax = times[peak_ids[0]]
                stretch = (colour_tmax - tmax) / 30
                stretch_err = tmax_err / 30  # inaccurate error propagation
            else:
                # warnings.warn(f"The peak in the colour curve was not found")
                stretch = stretch_err = np.nan

        return stretch, stretch_err

    def get_lc_params(self):
        r"""Calculates the peak magnitude (:math:`m_{max}`), its epoch
        (:math:`t_{max}`) and :math:`\Delta m_{15}` for all bands.
        """
        for band in self.bands:
            self[band].get_max()
            self[band].get_dm15()
