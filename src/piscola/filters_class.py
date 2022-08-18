import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import piscola


class SingleFilter(object):
    """Single filter class."""

    def __init__(self, band):
        """
        Parameters
        ----------
        band: str
            Name of the band.
        """
        self.name = band
        self._add_filter(band)

    def __repr__(self):
        rep = (
            f"name: {self.name}, eff_wave: {self.eff_wave:.1f} Å,"
            f" response_type: {self.response_type}"
        )
        return rep

    def __getitem__(self, item):
        return getattr(self, item)

    def _add_filter(self, filt_name):
        """Adds a filter from the available filters
        in the PISCOLA library.

        Parameters
        ----------
        filt_name: str
            Name of the filter.
        """
        pisco_path = piscola.__path__[0]
        filt_pattern = os.path.join(pisco_path, "filters", "*", f"{filt_name}.dat")
        filt_file = glob.glob(filt_pattern, recursive=True)

        err_message = (
            "No filter file or multiple files with "
            f"the pattern {filt_pattern} found."
        )
        assert len(filt_file) == 1, err_message

        filt_file = filt_file[0]
        self.filt_file = filt_file

        wave, transmission = np.loadtxt(filt_file).T
        # remove long tails of zero values on both edges
        imin, imax = trim_filters(transmission)
        self.wave = wave[imin:imax]
        self.transmission = transmission[imin:imax]

        # retrieve response type; if none, assumed to be photon type
        filt_dir = os.path.dirname(filt_file)
        resp_file = os.path.join(filt_dir, "response_type.txt")
        if os.path.isfile(resp_file):
            with open(resp_file) as file:
                line = file.readlines()[0]
                self.response_type = line.split()[0].lower()
        else:
            self.response_type = "photon"

        err_message = (
            f'"{self.response_type}" is not a valid response type '
            f'"photon" or "energy") for {filt_name} filter.'
        )
        assert self.response_type in ["photon", "energy"], err_message

        self.eff_wave = self.calc_eff_wave()

        readme_file = os.path.join(filt_dir, "README.txt")
        if os.path.isfile(readme_file):
            with open(readme_file, "r") as file:
                self.comments = file.read()
        else:
            self.comments = ""

    def calc_eff_wave(self, sed_wave=None, sed_flux=None):
        """Calculates the effective wavelength.

        The effective wavelength depends on the shape
        of the given SED.

        Parameters
        ----------
        sed_wave : array-like, default ``None``
            SED's wavelength range. If ``None``, a flat SED is used.
        sed_flux : array-like, default ``None``
            SED's flux density distribution. If ``None``, a flat SED is used.

        Returns
        -------
        eff_wave: float
            Effective wavelength.
        """
        if sed_wave is None or sed_flux is None:
            sed_wave = self.wave.copy()
            sed_flux = 100 * np.ones_like(sed_wave)

        transmission = self.transmission.copy()
        # check filter response type
        if self.response_type == "energy":
            transmission /= self.wave

        transmission = np.interp(sed_wave, self.wave, transmission, left=0.0, right=0.0)
        I1 = np.trapz((sed_wave**2) * transmission * sed_flux, sed_wave)
        I2 = np.trapz(sed_wave * transmission * sed_flux, sed_wave)
        eff_wave = I1 / I2

        return eff_wave

    def integrate_filter(self, sed_wave, sed_flux):
        """Calculates the flux density of an SED given a filter response.

        Parameters
        ----------
        sed_wave : array-like
            SED's wavelength range.
        sed_flux : array-like
            SED's flux density distribution.

        Returns
        -------
        flux_filter : float
            Flux density.
        """
        blue_edge_covered = sed_wave.min() <= self.wave.min()
        red_edge_covered = sed_wave.max() >= self.wave.max()
        err_message = "The SED does not completely overlap with {self.band} filter."
        assert blue_edge_covered and red_edge_covered, err_message

        transmission = self.transmission.copy()
        # check filter response type
        if self.response_type == "energy":
            transmission /= self.wave

        transmission = np.interp(sed_wave, self.wave, transmission, left=0.0, right=0.0)
        I1 = np.trapz(sed_flux * transmission * sed_wave, sed_wave)
        I2 = np.trapz(self.transmission * self.wave, self.wave)
        flux_filter = I1 / I2

        return flux_filter

    def _get_standard_flux(self, mag_sys_file):
        """Calculates the integrated flux of a standard star SED.

        Parameters
        ----------
        mag_sys_file: str
            File with the magnitude system information.

        Returns
        -------
        f_sed: float
            Integrated flux density.
        """
        pisco_path = piscola.__path__[0]

        # get standard SED file name
        with open(mag_sys_file, "rt") as file:
            for line in file:
                if len(line) > 0:
                    if "standard_sed:" in line.split():
                        standard_sed = line.split()[1]
                        break

        assert (
            "standard_sed" in locals()
        ), f"Standard SED file not found in {mag_sys_file}"

        if standard_sed.lower() == "ab":
            c = 2.99792458e18  # speed of light in [Å/s]
            sed_wave = np.arange(1000, 250000, 1)  # in [Å]
            sed_flux = 3631e-23 * c / sed_wave**2  # in [erg s^-1 cm^-2 Å^-1]
        else:
            sed_file = os.path.join(pisco_path, "standards", standard_sed)
            sed_wave, sed_flux = np.loadtxt(sed_file).T
        f_sed = self.integrate_filter(sed_wave, sed_flux)

        return f_sed

    def _get_standard_mag(self, mag_sys_file):
        """Obtains the tabulated magnitude of a standard star.

        Parameters
        ----------
        mag_sys_file: str
            File with the magnitude system information.

        Returns
        -------
        sed_mag: float
            Tabulated standard magnitude.
        """
        filt_names, mags = np.loadtxt(mag_sys_file, dtype=str).T
        err_message = f"{self.name} not in {mag_sys_file}"
        assert self.name in filt_names, err_message

        ind = list(filt_names).index(self.name)
        sed_mag = eval(mags[ind])

        return sed_mag

    def calc_zp(self, mag_sys):
        """Calculates the zero point in the ``AB``, ``Vega`` or ``BD17``
        magnitude systems.

        Parameters
        ----------
        mag_sys : str
            Magnitude system. For example: ``AB``, ``BD17`` or ``Vega``.

        Returns
        -------
        zp : float
            Zero-point in the given natural magnitude system.
        """
        pisco_path = piscola.__path__[0]
        mag_sys_file_path = os.path.join(pisco_path, "mag_sys", "magnitude_systems.txt")

        mag_sys_names, mag_sys_files = np.loadtxt(mag_sys_file_path, dtype=str).T
        err_message = f"mag. system {mag_sys} not found in {mag_sys_file_path}"
        assert mag_sys in mag_sys_names, err_message

        ind = list(mag_sys_names).index(mag_sys)
        mag_sys_file = os.path.join(pisco_path, "mag_sys", mag_sys_files[ind])

        f_sed = self._get_standard_flux(mag_sys_file)
        m_sed = self._get_standard_mag(mag_sys_file)
        zp = 2.5 * np.log10(f_sed) + m_sed

        return zp


class MultiFilters(object):
    """Class representing multiple filters."""

    def __init__(self, bands=None):
        """
        Parameters
        ----------
        bands: list-like
            Bands to include. If ``None``, only include
            Bessell filters.
        """
        if bands is None:
            bands = []
        self.bands = list(bands).copy()

        for band in bands:
            single_filt = SingleFilter(band)
            setattr(self, band, single_filt)

        # add Bessell filters
        filters = "UBVRI"
        for filt in filters:
            band = f"Bessell_{filt}"
            self.add_filter(band)

    def __repr__(self):
        return str(self.bands)

    def __getitem__(self, item):
        return getattr(self, item)

    def add_filter(self, band):
        """Adds a band object with a single-filter class.

        Parameters
        ----------
        band: str
            Name of the band.
        """
        single_filt = SingleFilter(band)
        setattr(self, band, single_filt)
        self.bands.append(band)

    def remove_filter(self, band):
        """Removes a band object.

        Parameters
        ----------
        band: str
            Name of the band.
        """
        err_message = f"Filter not found: {self.bands}"
        assert band in self.bands, err_message

        delattr(self, band)
        self.bands.remove(band)

    def calc_eff_wave(self, bands=None, sed_wave=None, sed_flux=None):
        """Calculates the effective wavelength of multiple bands.

        The effective wavelength depends on the shape
        of the given SED.

        Parameters
        ----------
        bands: list-like, default ``None``
            Bands for calculating their effective wavelengths. If ``None``,
            use of the available bands.
        sed_wave : array-like, default ``None``
            SED's wavelength range. If ``None``, a flat SED is used.
        sed_flux : array-like, default ``None``
            SED's flux density distribution. If ``None``, a flat SED is used.
        """
        if not bands:
            bands = self.bands

        for band in bands:
            self[band].calc_eff_wave(sed_wave, sed_flux)

    def calc_pivot(self, z=0.0):
        """Calculates the observed band closest to restframe
        :math:`B` band (:math:`4500 Å`).

        Parameters
        ----------
        z : float, default ``0.0``
            Redshift.
        """
        B_eff_wave = 4500.0
        eff_waves = np.array([self[band].eff_wave / (1 + z) for band in self.bands])
        idx = (np.abs(B_eff_wave - eff_waves)).argmin()
        self.pivot_band = self.bands[idx]

    def plot_filters(self, bands=None):
        """Plot the filters' transmission functions.

        Parameters
        ----------
        bands : list, default ``None``
            Bands to plot. If ``None``, plots all the available bands.
        """
        if not bands:
            bands = self.bands

        fig, ax = plt.subplots(figsize=(8, 6))
        for band in bands:
            norm = self[band]["transmission"].max()
            ax.plot(self[band]["wave"], self[band]["transmission"] / norm, label=band)

        ax.set_xlabel(r"Wavelength ($\AA$)", fontsize=18, family="serif")
        ax.set_ylabel("Normalised Transmission Function", fontsize=18, family="serif")
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.minorticks_on()
        ax.tick_params(
            which="major", length=6, width=1, direction="in", top=True, right=True
        )
        ax.tick_params(
            which="minor", length=3, width=1, direction="in", top=True, right=True
        )
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=14)
        # plt.tight_layout()
        # plt.savefig('filters.png')

        plt.show()


def trim_filters(response):
    """Trim the leading and trailing zeros from a 1-D array or sequence, leaving
    one zero on each side. This is a modified version of :func:`numpy.trim_zeros`.

    Parameters
    ----------
    response : 1-D array or sequence
        Input array.

    Returns
    -------
    first : int
        Index of the last leading zero.
    last : int
        Index of the first trailing zero.
    """

    first = 0
    for i in response:
        if i != 0.0:
            if first == 0:
                first += 1  # to avoid filters with non-zero edges
            break
        else:
            first = first + 1

    last = len(response)
    for i in response[::-1]:
        if i != 0.0:
            if last == len(response):
                last -= 1  # to avoid filters with non-zero edges
            break
        else:
            last = last - 1

    first -= 1
    last += 1

    return first, last
