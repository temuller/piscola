import numpy as np

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
    if type(flux)==np.ndarray:
        # turns negative and 0.0 values into NaNs
        flux_ = np.array([f if f>=0.0 else np.nan for f in flux])
    elif flux<=0.0:
        flux_ = np.nan
    else:
        flux_ = flux
        
    mag = -2.5*np.log10(flux_) + zp
    mag_err = np.abs( 2.5*flux_err/(flux_*np.log(10)) )

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
    flux = 10**( -0.4*(mag-zp) )
    flux_err =  np.abs( flux*0.4*np.log(10)*mag_err )

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
    new_flux = flux*10**( -0.4*(zp - new_zp) )

    return new_flux