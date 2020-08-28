import numpy as np
import piscola
#from astropy import units as u
#from astropy import constants as const

#h = const.cgs.codata2014.h.value
#c = const.c.to(u.AA/u.s).value

def integrate_filter(spectrum_wave, spectrum_flux, filter_wave, filter_response, response_type='photon'):
    """Calcultes the flux density of an SED given a filter response.

    Parameters
    ----------
    spectrum_wave : array
        Spectrum's wavelength range.
    spectrum_flux : array
        Spectrum's flux density distribution.
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    response_type : str, default 'photon'
        Filter's response type. Either 'photon' or 'energy'. Only the Bessell filters use 'energy'.

    Returns
    -------
    Flux density.

    """

    # truncate the filter at both sides by the same amount if it is not not cover by most of the spectrum range
    imax = np.argmin(np.abs(spectrum_wave.max() - filter_wave)) + 1
    imin = np.argmin(np.abs(spectrum_wave.min() - filter_wave))

    if imax != len(filter_wave):
        imin = len(filter_wave) - imax
    elif imin != 0:
        imax = len(filter_wave) - imin

    min_index, max_index = filter_effective_range(filter_response)
    assert (imin <= min_index) and (imax >= max_index), 'The spectrum does not cover enough range of the filter used.'

    filter_wave = filter_wave[imin:imax]
    filter_response = filter_response[imin:imax]

    #check filter response type
    if response_type == 'energy':
        filter_response = filter_response/filter_wave

    interp_response = np.interp(spectrum_wave, filter_wave, filter_response, left=0.0, right=0.0)
    I1 = np.trapz(spectrum_flux*interp_response*spectrum_wave, spectrum_wave)
    I2 = np.trapz(filter_response*filter_wave, filter_wave)
    flux_filter = I1/I2

    return flux_filter


def calc_eff_wave(spectrum_wave, spectrum_flux, filter_wave, filter_response, response_type='photon'):
    """Calcultes the effective wavelength of the filter given an SED.

    Parameters
    ----------
    spectrum_wave : array
        Spectrum's wavelength range.
    spectrum_flux : array
        Spectrum's flux density distribution.
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    response_type : str, default 'photon'
        Filter's response type. Either 'photon' or 'energy'. Only the Bessell filters use 'energy'.

    Returns
    -------
    Filter's effective wavelength.

    """

    #check filter response type
    if response_type == 'energy':
        filter_response = filter_response/filter_wave

    interp_response = np.interp(spectrum_wave, filter_wave, filter_response, left=0.0, right=0.0)
    I1 = np.trapz((spectrum_wave**2)*interp_response*spectrum_flux, spectrum_wave)
    I2 = np.trapz(spectrum_wave*interp_response*spectrum_flux, spectrum_wave)
    eff_wave = I1/I2

    return eff_wave


def calc_pivot_wave(filter_wave, filter_response, response_type):
    """Calcultes the pivot wavelength for the given filter.

    Parameters
    ----------
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    response_type : str, default 'photon'
        Filter's response type. Either 'photon' or 'energy'. Only the Bessell filters use 'energy'.

    Returns
    -------
    Filter's pivot wavelength.

    """

    #check filter response type
    if response_type == 'energy':
        filter_response = filter_response/filter_wave

    I1 = np.trapz(filter_response*filter_wave, filter_wave)
    I2 = np.trapz(filter_response/filter_wave, filter_wave)
    pivot_wave = np.sqrt(I1/I2)

    return pivot_wave


def calc_zp(filter_wave, filter_response, response_type, mag_sys, filter_name, offsets_file=None):
    """Calculates the zero point in the AB, Vega or BD17 magnitude sytems.

    Parameters
    ----------
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    response_type : str
        Filter's response type. Either 'photon' or 'energy'. Only the Bessell filters use 'energy'.
    mag_sys : str
        Magnitude system. Either 'AB', 'Vega', 'BD17'.
    filter_name : str
        Filter name. Used to estimate the zero point for the 'BD17' magnitude system.

    Returns
    -------
    Zero-point in the Vega magnitude system.

    """

    path = piscola.__path__[0]

    if offsets_file:
        file_path = f'{path}/templates/{offsets_file}'
    else:
        file_path = f'{path}/templates/{mag_sys.lower()}_sys_zps.dat'

    if mag_sys.lower() == 'ab':
        c = 2.99792458e18  # speed of light in [Angstroms/s]
        ab_wave = np.arange(1000, 250000, 5)
        ab_flux = 3631e-23*c/ab_wave**2  # in [erg s^-1 cm^-2 A^-1]
        f_ab = integrate_filter(ab_wave, ab_flux, filter_wave, filter_response, response_type)

        # get ZP offsets
        with open(file_path, 'rt') as ab_file:
            ab_mag = [line.split() for line in ab_file if filter_name in line.split()]
        if ab_mag:
            zp = 2.5*np.log10(f_ab) + eval(ab_mag[0][-1])
        else:
            raise ValueError(f'Could not find "{filter_name}" filter in {file_path}')

    elif mag_sys.lower() == 'vega':
        spectrum_wave, spectrum_flux = np.loadtxt(path + '/templates/alpha_lyr_stis_005.dat').T
        f_vega = integrate_filter(spectrum_wave, spectrum_flux, filter_wave, filter_response, response_type)
        zp = 2.5*np.log10(f_vega)

    elif mag_sys.lower() == 'bd17':
        spectrum_wave, spectrum_flux = np.loadtxt(path + '/templates/bd_17d4708_stisnic_005.dat').T
        f_bd17 = integrate_filter(spectrum_wave, spectrum_flux, filter_wave, filter_response, response_type)

        # get ZP offsets
        with open(file_path, 'rt') as bd17_file:
            bd17_mag = [line.split() for line in bd17_file if filter_name in line.split()]
        if bd17_mag:
            zp = 2.5*np.log10(f_bd17) +  eval(bd17_mag[0][-1])
        else:
            raise ValueError(f'Could not find "{filter_name}" filter in {file_path}')
    else:
        raise ValueError(f'Could not find "{mag_sys}" magnitude system')

    return zp


def filter_effective_range(filter_response, percent=99.0):
    """Finds the min and max indexes which contain at least the desire percentage of the filter's
    response-function area.

    Parameters
    ----------
    filter_response : array
        Filter's response function.
    percent : float, default '99.0'
        Percentage of the filter's area that wants to be kept.

    Returns
    -------
    Minimum and maximum indexes which contain the wanted area of the filter, independently.

    """

    for min_index in range(len(filter_response)):
        max_index = len(filter_response) - min_index
        area = 100*np.trapz(filter_response[min_index:max_index])/np.trapz(filter_response)
        if area < percent:
            break

    # to prevent going beyond the edges of the array
    if min_index == 0:
        min_index += 1
        max_index -= 1

    return min_index-1, max_index+1
