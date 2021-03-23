import piscola
from .filter_utils import integrate_filter

import extinction
import sfdmap

import matplotlib.pyplot as plt
import numpy as np
import os

def redden(wave, flux, ra, dec, scaling=0.86, reddening_law='fitzpatrick99'):
    """Reddens the given spectrum, given a right ascension and declination. $R_V$ is assumed to be 3.1.

    Parameters
    ----------
    wave : array
        Wavelength values.
    flux : array
        Flux density values.
    ra : float
        Right ascension.
    dec : float
        Declination in degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law.``fitzpatrick99`` for Fitzpatrick99 (1999) or ``ccm89`` for Cardelli, Clayton & Mathis (1989).

    Returns
    -------
    redden_flux : array
        Redden flux values.

    """

    path = piscola.__path__[0]
    mapdir = os.path.join(path, 'sfddata-master')
    m = sfdmap.SFDMap(mapdir=mapdir, scaling=scaling)
    ebv = m.ebv(ra, dec) # RA and DEC in degrees
    r_v  = 3.1
    a_v = r_v*ebv

    if reddening_law=='fitzpatrick99':
        ext = extinction.fitzpatrick99(wave, a_v, r_v)
    elif reddening_law=='ccm89':
        ext = extinction.ccm89(wave, a_v, r_v)
    redden_flux = extinction.apply(ext, flux)

    return redden_flux


def deredden(wave, flux, ra, dec, scaling=0.86, reddening_law='fitzpatrick99'):
    """Dereddens the given spectrum, given a right ascension and declination. Rv is assumed to be 3.1.

    Parameters
    ----------
    wave : array
        Wavelength values.
    flux : array
        Flux density values.
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, ``Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law.``fitzpatrick99`` for Fitzpatrick99 (1999) or ``ccm89`` for Cardelli, Clayton & Mathis (1989).

    Returns
    -------
    deredden_flux : array
        Deredden flux values.
    Returns the deredden flux density values.

    """

    path = piscola.__path__[0]
    mapdir = os.path.join(path, 'sfddata-master')
    m = sfdmap.SFDMap(mapdir=mapdir, scaling=scaling)
    ebv = m.ebv(ra, dec) # RA and DEC in degrees
    r_v  = 3.1
    a_v = r_v*ebv

    if reddening_law=='fitzpatrick99':
        ext = extinction.fitzpatrick99(wave, a_v, r_v)
    elif reddening_law=='ccm89':
        ext = extinction.ccm89(wave, a_v, r_v)
    deredden_flux = extinction.remove(ext, flux)

    return deredden_flux


def calculate_ebv(ra, dec, scaling=0.86):
    """Calculates Milky Way reddening, $E(B-V)$.

    Parameters
    ----------
    ra : float
        Right ascension.
    dec : float
        Declination
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Finkbeiner & Davis (1998).

    Returns
    -------
    ebv :  float
        Reddening value, $E(B-V)$.

    """

    path = piscola.__path__[0]
    mapdir = os.path.join(path, 'sfddata-master')
    m = sfdmap.SFDMap(mapdir=mapdir, scaling=scaling)
    ebv = m.ebv(ra, dec) # RA and DEC in degrees

    return ebv


def extinction_filter(filter_wave, filter_response, ra, dec, scaling=0.86, reddening_law='fitzpatrick99'):
    """Estimate the extinction for a given filter, given a right ascension and declination. $R_V$ is assumed to be 3.1.

    Parameters
    ----------
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    ra : float
        Right ascension.
    dec : float
        Declinationin degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, ``Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law.``fitzpatrick99`` for Fitzpatrick99 (1999) or ``ccm89`` for Cardelli, Clayton & Mathis (1989).

    Returns
    -------
    A : float
        Extinction value in magnitudes.

    """

    flux = 100
    deredden_flux = deredden(filter_wave, flux, ra, dec, scaling, reddening_law)

    f1 = integrate_filter(filter_wave, flux, filter_wave, filter_response)
    f2 = integrate_filter(filter_wave, deredden_flux, filter_wave, filter_response)
    A = -2.5*np.log10(f1/f2)

    return A


def extinction_curve(ra, dec, scaling=0.86, reddening_law='fitzpatrick99'):
    """Plots the extinction curve for a given RA and Dec. $R_V$ is assumed to be 3.1.

    Parameters
    ----------
    ra : float
        Right ascension.
    dec : float
        Declination in degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, ``Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law.``fitzpatrick99`` for Fitzpatrick99 (1999) or ``ccm89`` for Cardelli, Clayton & Mathis (1989).

    """

    flux = 100
    wave = np.arange(1000, 25001)  # in Angstroms
    deredden_flux = deredden(wave, flux, ra, dec, scaling, reddening_law)
    ff = 1 - flux/deredden_flux

    f, ax = plt.subplots(figsize=(8,6))
    ax.plot(wave, ff)

    ax.set_xlabel(r'wavelength ($\AA$)', fontsize=18)
    ax.set_ylabel('fraction of extinction', fontsize=18)
    ax.set_title(r'Extinction curve', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)

    plt.show()
