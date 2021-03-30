import os
import numpy as np
import piscola

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
    response_type : str, default ``photon``
        Filter's response type. Either ``photon`` or ``energy``.

    Returns
    -------
    flux_filter : float
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
        filter_response = filter_response.copy()/filter_wave

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
    response_type : str, default ``photon``
        Filter's response type. Either ``photon`` or ``energy``.

    Returns
    -------
    eff_wave : float
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


def calc_pivot_wave(filter_wave, filter_response, response_type='photon'):
    """Calcultes the pivot wavelength for the given filter.

    Parameters
    ----------
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    response_type : str, default ``photon``
        Filter's response type. Either ``photon`` or ``energy``.

    Returns
    -------
    pivot_wave : float
        Filter's pivot wavelength.

    """

    #check filter response type
    if response_type == 'energy':
        filter_response = filter_response/filter_wave

    I1 = np.trapz(filter_response*filter_wave, filter_wave)
    I2 = np.trapz(filter_response/filter_wave, filter_wave)
    pivot_wave = np.sqrt(I1/I2)

    return pivot_wave


def calc_zp(filter_wave, filter_response, response_type, mag_sys, filter_name):
    """Calculates the zero point in the AB, Vega or BD17 magnitude systems.

    Parameters
    ----------
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    response_type : str, default ``photon``
        Filter's response type. Either ``photon`` or ``energy``.
    mag_sys : str
        Magnitude system. For example, ``AB``, ``BD17`` or ``Vega``.
    filter_name : str
        Filter name.

    Returns
    -------
    zp : float
        Zero-point in the given natural magnitude system.

    """

    path = piscola.__path__[0]
    mag_sys_dict = {}
    mag_sys_file_path = os.path.join(path, 'standards/magnitude_systems.txt')
    with open(mag_sys_file_path) as mag_sys_file:
        for line in mag_sys_file:
            (key, val) = line.split()  # key:magnitude system name, val: file with natural system values
            mag_sys_dict[key] = val

    assert mag_sys.upper() in mag_sys_dict.keys(), f"magnitude system '{mag_sys.upper()}' not found in '{mag_sys_file_path}'"

    file_path = os.path.join(path, 'standards', mag_sys_dict[mag_sys.upper()])

    if ('ab' in mag_sys.split('_')) or ('AB' in mag_sys.split('_')):
        c = 2.99792458e18  # speed of light in [Angstroms/s]
        ab_wave = np.arange(1000, 250000, 5)
        ab_flux = 3631e-23*c/ab_wave**2  # in [erg s^-1 cm^-2 A^-1]
        f_ab = integrate_filter(ab_wave, ab_flux, filter_wave, filter_response, response_type)

        # get ZP offsets
        with open(file_path, 'rt') as ab_sys_file:
            ab_mag = [line.split() for line in ab_sys_file if filter_name in line.split()]
        if ab_mag:
            zp = 2.5*np.log10(f_ab) + eval(ab_mag[0][1])
        else:
            raise ValueError(f'Could not find "{filter_name}" filter in {file_path}')

    elif 'vega' in mag_sys.lower():
        vega_sed_file = os.path.join(path, 'standards/alpha_lyr_stis_005.dat')
        spectrum_wave, spectrum_flux = np.loadtxt(vega_sed_file).T
        f_vega = integrate_filter(spectrum_wave, spectrum_flux, filter_wave, filter_response, response_type)
        zp = 2.5*np.log10(f_vega)

    elif 'bd17' in mag_sys.lower():
        # get ZP offsets
        with open(file_path, 'rt') as bd17_sys_file:
            standard_sed = [line.split()[1] for line in bd17_sys_file if 'standard_sed:' in line.split()][0]

        bd17_sed_file = os.path.join(path, 'standards', standard_sed)
        spectrum_wave, spectrum_flux = np.loadtxt(bd17_sed_file).T
        f_bd17 = integrate_filter(spectrum_wave, spectrum_flux, filter_wave, filter_response, response_type)

        with open(file_path, 'rt') as bd17_sys_file:
            bd17_mag = [line.split() for line in bd17_sys_file if filter_name in line.split()]

        if bd17_mag:
            zp = 2.5*np.log10(f_bd17) +  eval(bd17_mag[0][1])
        else:
            raise ValueError(f'Could not find "{filter_name}" filter in {file_path}')
    else:
        raise ValueError(f'Could not find "{mag_sys}" magnitude system in the implemented systems of the code')

    return zp


def filter_effective_range(filter_response, percent=99.0):
    """Finds the min and max indexes which contain at least the desire percentage of the filter's
    response-function area.

    **Note:** each index contains the wanted area independently from the other.

    Parameters
    ----------
    filter_response : array
        Filter's response function.
    percent : float, default ``99.0``
        Percentage of the filter's area that wants to be kept.

    Returns
    -------
    min_index-1 : int
        Minimum index containing the wanted area of the filter.
    max_index+1 : int
        Maximum index containing the wanted area of the filter.

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
