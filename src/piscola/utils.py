import numpy as np


def _integrate_filter(
    sed_wave, sed_flux, filter_wave, filter_transmission, response_type="photon"
):
    """Calculates the flux density of a spectral energy distribution (SED)
    given a filter transmission function.

    Mainly used by the :func:`extinction_correction` module.

    Parameters
    ----------
    sed_wave : array
        SED's wavelength range.
    sed_flux : array
        SED's flux density distribution.
    filter_wave : array
        Filter's wavelength range.
    filter_transmission : array
        Filter's transmission function.
    response_type : str, default ``photon``
        Filter's response type. Either ``photon`` or ``energy``.

    Returns
    -------
    flux_filter : float
        Flux density.
    """

    blue_edge_covered = sed_wave.min() <= filter_wave.min()
    red_edge_covered = sed_wave.max() >= filter_wave.max()
    err_message = "The SED does not completely overlap with {self.band} filter."
    assert blue_edge_covered and red_edge_covered, err_message

    transmission = filter_transmission.copy()
    # check filter response type
    if response_type == "energy":
        transmission /= filter_wave

    transmission = np.interp(sed_wave, filter_wave, transmission, left=0.0, right=0.0)
    I1 = np.trapz(sed_flux * transmission * sed_wave, sed_wave)
    I2 = np.trapz(filter_transmission * filter_wave, filter_wave)
    flux_filter = I1 / I2

    return flux_filter


def flux2mag(flux, zp, flux_err=0.0):
    """Converts fluxes to magnitudes, propagating errors if given.

    Note: if there are negative or zero fluxes, these are converted to NaN values.

    Parameters
    ----------
    flux : array
        Array of fluxes.
    zp : float or array
        Zero points.
    flux_err : array, default ``0.0``
        Array of flux errors.

    Returns
    -------
    mag : array
        Fluxes converted to magnitudes.
    mag_err : array
        Flux errors converted to errors in magnitudes.
    """
    flux_ = flux.copy()
    flux_[flux_ <= 0.0] = np.nan

    flux_err_ = flux_err.copy()
    flux_err_[flux_err_ <= 0.0] = np.nan

    mag = -2.5 * np.log10(flux_) + zp
    mag_err = np.abs(2.5 * flux_err_ / (flux_ * np.log(10)))

    return mag, mag_err


def mag2flux(mag, zp, mag_err=0.0):
    """Converts magnitudes to fluxes, propagating errors if given.

    Parameters
    ----------
    mag : array
        Array of magnitudes.
    zp : float or array
        Zero points.
    mag_err : array, default ``0.0``
        Array of magnitude errors.

    Returns
    -------
    flux : array
        Magnitudes converted to fluxes.
    flux_err : array
        Magnitude errors converted to errors in fluxes.
    """
    mag_ = mag.copy()
    mag_[mag_ <= 0.0] = np.nan

    mag_err_ = mag_err.copy()
    mag_err_[mag_err_ <= 0.0] = np.nan

    flux = 10 ** (-0.4 * (mag_ - zp))
    flux_err = np.abs(flux * 0.4 * np.log(10) * mag_err_)

    return flux, flux_err


def change_zp(flux, zp, new_zp):
    """Converts flux units given a new zero-point.

    **Note:** this assumes that the new zero-point is in the same magnitude system as the current one.

    Parameters
    ----------
    flux : float or array
        Fluxes.
    zp : float or array
        Current zero-point for the given fluxes.
    new_zp : float or array
        New zero-point to convert the flux units.

    Returns
    -------
    new_flux : float or array
        Fluxes with with a new zero-point.
    """
    new_flux = flux * 10 ** (-0.4 * (zp - new_zp))

    return new_flux
