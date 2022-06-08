import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import piscola

class single_filter(object):
    """Single filter class.
    """
    def __init__(self, band):
        self.name = band
        self._sed = None
        self.add_filter(band)

    def __repr__(self):
        rep = (f'name: {self.name}, eff_wave: {self.eff_wave:.1f} Å,'
               f' response_type: {self.response_type}')
        return rep

    def __getitem__(self, item):
        return getattr(self, item)

    def add_filter(self, filt_name):
        """Add a filter from the available filters
        in the PISCOLA library.
        """
        pisco_path = piscola.__path__[0]
        filt_pattern = os.path.join(pisco_path, 'filters',
                                    '*', f'{filt_name}.dat')
        filt_file = glob.glob(filt_pattern, recursive=True)

        err_message = ('No filter file or multiple files with '
                       f'the pattern {filt_pattern} found.')
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
        resp_file = os.path.join(filt_dir, 'response_type.txt')
        if os.path.isfile(resp_file):
            with open(resp_file) as file:
                line = file.readlines()[0]
                self.response_type = line.split()[0].lower()
        else:
            self.response_type = 'photon'

        err_message = (f'"{self.response_type}" is not a valid response type '
                       f'"photon" or "energy") for {filt_name} filter.')
        assert self.response_type in ['photon', 'energy'], err_message

        self.eff_wave = self.calc_eff_wave()

        readme_file = os.path.join(filt_dir, 'README.txt')
        if os.path.isfile(readme_file):
            with open(readme_file, 'r') as file:
                self.comments = file.read()
        else:
            self.comments = ''

    def calc_eff_wave(self, sed_wave=None, sed_flux=None):

        if not sed_wave or not sed_flux:
            sed_wave = self.wave.copy()
            sed_flux = 100 * np.ones_like(sed_wave)

        transmission = self.transmission.copy()
        # check filter response type
        if self.response_type == 'energy':
            transmission /= self.wave

        transmission = np.interp(sed_wave, self.wave, transmission,
                                 left=0.0, right=0.0)
        I1 = np.trapz((sed_wave ** 2) * transmission * sed_flux, sed_wave)
        I2 = np.trapz(sed_wave * transmission * sed_flux, sed_wave)
        eff_wave = I1 / I2

        return eff_wave

    def integrate_filter(self, sed_wave, sed_flux):
        """Calculates the flux density of an SED given a filter response.

        Parameters
        ----------
        sed_wave : array
            Spectrum's wavelength range.
        sed_flux : array
            Spectrum's flux density distribution.

        Returns
        -------
        flux_filter : float
            Flux density.
        """
        blue_edge_covered = sed_wave.min() <= self.wave.min()
        red_edge_covered = sed_wave.max() >= self.wave.max()
        err_message = 'The SED does not completely overlap with {self.band} filter.'
        assert blue_edge_covered and red_edge_covered, err_message

        transmission = self.transmission.copy()
        # check filter response type
        if self.response_type == 'energy':
            ftransmission /= self.wave

        transmission = np.interp(sed_wave, self.wave, transmission,
                                 left=0.0, right=0.0)
        I1 = np.trapz(sed_flux * transmission * sed_wave, sed_wave)
        I2 = np.trapz(self.transmission * self.wave, self.wave)
        flux_filter = I1 / I2

        return flux_filter

class multi_filters(object):
    """Class representing multiple filters.
    """
    def __init__(self, bands):
        self.bands = bands

        for band in bands:
            single_filt = single_filter(band)
            setattr(self, band, single_filt)

        # add Bessell filters
        filters = 'UBVRI'
        for filt in filters:
            band = f'Bessell_{filt}'
            single_filt = single_filter(band)
            setattr(self, band, single_filt)

    def __repr__(self):
        return str(self.bands)

    def __getitem__(self, item):
        return getattr(self, item)

    def add_filter(self, band):
        single_filt = single_filter(band)
        setattr(self, band, single_filt)
        self.bands.append(band)

    def remove_filter(self, band):
        err_message = f'Filter not found: {self.bands}'
        assert band in self.bands, err_message

        delattr(self, band)
        self.bands.remove(band)

    def calc_eff_wave(self, bands=None, sed_wave=None, sed_flux=None):
        if not bands:
            bands = self.bands

        for band in bands:
            self[band].calc_eff_wave(sed_wave, sed_flux)

    def plot_filters(self, bands=None):
        """Plot the filters' transmission functions.

        Parameters
        ----------
        bands : list, default ``None``
            List of bands.
        """
        if not bands:
            bands = self.bands

        fig, ax = plt.subplots(figsize=(8,6))
        for band in bands:
            norm = self[band]['transmission'].max()
            ax.plot(self[band]['wave'], self[band]['transmission']/norm, label=band)

        ax.set_xlabel(r'wavelength ($\AA$)', fontsize=18, family='serif')
        ax.set_ylabel('normalized response', fontsize=18, family='serif')
        ax.set_title(r'Filters response functions', fontsize=18, family='serif')
        ax.xaxis.set_tick_params(labelsize=15)
        ax.yaxis.set_tick_params(labelsize=15)
        ax.minorticks_on()
        ax.tick_params(which='major', length=6, width=1, direction='in', top=True, right=True)
        ax.tick_params(which='minor', length=3, width=1, direction='in', top=True, right=True)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

        plt.show()

def trim_filters(response):
    """Trim the leading and trailing zeros from a 1-D array or sequence, leaving
    one zero on each side. This is a modified version of numpy.trim_zeros.

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
        if i != 0.:
            if first == 0:
                first += 1  # to avoid filters with non-zero edges
            break
        else:
            first = first + 1

    last = len(response)
    for i in response[::-1]:
        if i != 0.:
            if last == len(response):
                last -= 1  # to avoid filters with non-zero edges
            break
        else:
            last = last - 1

    first -= 1
    last += 1

    return first, last
