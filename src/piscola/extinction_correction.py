import os
import tarfile
import requests
import numpy as np
import matplotlib.pyplot as plt

import sfdmap
import extinction

import piscola
from .utils import _integrate_filter

pisco_path = piscola.__path__[0]
dustmaps_dir = os.path.join(pisco_path, "sfddata-master")


def _download_dustmaps():
    """Downloads the dust maps for extinction calculation if they are not found
    locally.
    """
    global dustmaps_dir

    # check if files already exist locally
    dust_files = [
        os.path.join(dustmaps_dir, f"SFD_dust_4096_{sky}gp.fits") for sky in ["n", "s"]
    ]
    mask_files = [
        os.path.join(dustmaps_dir, f"SFD_mask_4096_{sky}gp.fits") for sky in ["n", "s"]
    ]
    maps_files = dust_files + mask_files
    existing_files = [os.path.isfile(file) for file in mask_files]

    if not all(existing_files) == True:
        # download dust maps
        sfdmaps_url = "https://github.com/kbarbary/sfddata/archive/master.tar.gz"
        response = requests.get(sfdmaps_url)

        master_tar = "master.tar.gz"
        with open(master_tar, "wb") as file:
            file.write(response.content)

        # extract tar file under mapsdir directory
        tar = tarfile.open(master_tar)
        tar.extractall(pisco_path)
        tar.close()
        os.remove(master_tar)


def redden(
    wave, flux, ra, dec, scaling=0.86, reddening_law="fitzpatrick99", r_v=3.1, ebv=None
):
    """Reddens the given spectrum, given a right ascension and declination or :math:`E(B-V)`.

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
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989), ``odonnell94`` (O’Donnell 1994),
        ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00`` (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with
        :math:`R_V = 3.1`.)
    r_v : float, default ``3.1``
        Total-to-selective extinction ratio (:math:`R_V`)
    ebv : float, default ``None``
        Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.

    Returns
    -------
    redden_flux : array
        Redden flux values.

    """
    _download_dustmaps()
    global dustmaps_dir

    if ebv is None:
        dustmaps_dir = os.path.join(pisco_path, "sfddata-master")
        m = sfdmap.SFDMap(mapdir=dustmaps_dir, scaling=scaling)
        ebv = m.ebv(ra, dec)  # RA and DEC in degrees

    a_v = r_v * ebv

    rl_list = ["ccm89", "odonnell94", "fitzpatrick99", "calzetti00", "fm07"]
    err_message = f"Choose one of the available reddening laws: {rl_list}"
    assert reddening_law in rl_list, err_message

    wave = np.copy(wave.astype("double"))
    if reddening_law == "ccm89":
        ext = extinction.ccm89(wave, a_v, r_v)
    elif reddening_law == "odonnell94":
        ext = extinction.odonnell94(wave, a_v, r_v)
    elif reddening_law == "fitzpatrick99":
        ext = extinction.fitzpatrick99(wave, a_v, r_v)
    elif reddening_law == "calzetti00":
        ext = extinction.calzetti00(wave, a_v, r_v)
    elif reddening_law == "fm07":
        ext = extinction.fm07(wave, a_v)

    redden_flux = extinction.apply(ext, flux)

    return redden_flux


def deredden(
    wave, flux, ra, dec, scaling=0.86, reddening_law="fitzpatrick99", r_v=3.1, ebv=None
):
    """Dereddens the given spectrum, given a right ascension and declination or :math:`E(B-V)`.

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
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989), ``odonnell94`` (O’Donnell 1994),
        ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00`` (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with
        :math:`R_V = 3.1`.)
    r_v : float, default ``3.1``
        Total-to-selective extinction ratio (:math:`R_V`)
    ebv : float, default ``None``
        Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.

    Returns
    -------
    deredden_flux : array
        Deredden flux values.

    """
    _download_dustmaps()
    global dustmaps_dir

    if ebv is None:
        m = sfdmap.SFDMap(mapdir=dustmaps_dir, scaling=scaling)
        ebv = m.ebv(ra, dec)  # RA and DEC in degrees

    a_v = r_v * ebv

    rl_list = ["ccm89", "odonnell94", "fitzpatrick99", "calzetti00", "fm07"]
    assert (
        reddening_law in rl_list
    ), f"Choose one of the available reddening laws: {rl_list}"

    if reddening_law == "ccm89":
        ext = extinction.ccm89(wave, a_v, r_v)
    elif reddening_law == "odonnell94":
        ext = extinction.odonnell94(wave, a_v, r_v)
    elif reddening_law == "fitzpatrick99":
        ext = extinction.fitzpatrick99(wave, a_v, r_v)
    elif reddening_law == "calzetti00":
        ext = extinction.calzetti00(wave, a_v, r_v)
    elif reddening_law == "fm07":
        ext = extinction.fm07(wave, a_v)

    deredden_flux = extinction.remove(ext, flux)

    return deredden_flux


def calculate_ebv(ra, dec, scaling=0.86):
    """Calculates Milky Way colour excess, :math:`E(B-V)`, for a given right ascension and declination.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Finkbeiner & Davis (1998).

    Returns
    -------
    ebv :  float
        Reddening value, :math:`E(B-V)``.
    """
    _download_dustmaps()
    global dustmaps_dir
    m = sfdmap.SFDMap(mapdir=dustmaps_dir, scaling=scaling)
    ebv = m.ebv(ra, dec)  # RA and DEC in degrees

    return ebv


def extinction_filter(
    filter_wave,
    filter_response,
    ra,
    dec,
    scaling=0.86,
    reddening_law="fitzpatrick99",
    r_v=3.1,
    ebv=None,
):
    """Estimate the extinction for a given filter, given a right ascension and declination or :math:`E(B-V)`.

    Parameters
    ----------
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989), ``odonnell94`` (O’Donnell 1994),
        ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00`` (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with
        :math:`R_V = 3.1`.)
    r_v : float, default ``3.1``
        Total-to-selective extinction ratio (:math:`R_V`)
    ebv : float, default ``None``
        Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.

    Returns
    -------
    A : float
        Extinction value in magnitudes.
    """
    flux = 100
    deredden_flux = deredden(
        filter_wave, flux, ra, dec, scaling, reddening_law, r_v, ebv
    )

    f1 = _integrate_filter(filter_wave, flux, filter_wave, filter_response)
    f2 = _integrate_filter(filter_wave, deredden_flux, filter_wave, filter_response)
    A = -2.5 * np.log10(f1 / f2)

    return A


def extinction_curve(
    ra, dec, scaling=0.86, reddening_law="fitzpatrick99", r_v=3.1, ebv=None
):
    """Plots the extinction curve for a given right ascension and declination or :math:`E(B-V)`.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    scaling: float, default ``0.86``
        Calibration of the Milky Way dust maps. Either ``0.86``
        for the Schlafly & Finkbeiner (2011) recalibration or ``1.0`` for the original
        dust map of Schlegel, Fikbeiner & Davis (1998).
    reddening_law: str, default ``fitzpatrick99``
        Reddening law. The options are: ``ccm89`` (Cardelli, Clayton & Mathis 1989), ``odonnell94`` (O’Donnell 1994),
        ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00`` (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007 with
        :math:`R_V = 3.1`.)
    r_v : float, default ``3.1``
        Total-to-selective extinction ratio (:math:`R_V`)
    ebv : float, default ``None``
        Colour excess (:math:`E(B-V)`). If given, this is used instead of the dust map value.

    """
    flux = 100
    wave = np.arange(1000, 25001)  # in Angstroms
    deredden_flux = deredden(wave, flux, ra, dec, scaling, reddening_law, r_v, ebv)
    ff = 1 - flux / deredden_flux

    f, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wave, ff)

    ax.set_xlabel(r"wavelength ($\AA$)", fontsize=18)
    ax.set_ylabel("fraction of extinction", fontsize=18)
    ax.set_title(r"Extinction curve", fontsize=18)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)

    plt.show()
