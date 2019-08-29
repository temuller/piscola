import numpy as np

def run_filter(spectrum_wave, spectrum_flux, filter_wave, filter_response, response_type='photon'):
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

    #check filter response type
    if response_type == 'photon':
        interp_response = np.interp(spectrum_wave, filter_wave, filter_response)
        I1, I2 = np.trapz(spectrum_flux*interp_response*spectrum_wave, spectrum_wave), np.trapz(filter_response*filter_wave, filter_wave)
        flux_filter = I1/I2
        
    elif response_type == 'energy':
        interp_response = np.interp(spectrum_wave, filter_wave, filter_response)
        I1, I2 = np.trapz(spectrum_flux*interp_response, spectrum_wave), np.trapz(filter_response, filter_wave)
        flux_filter = I1/I2
        
    else:
        raise ValueError(f'"{response_type}" is not a valis response type.')
        
    return flux_filter


def calc_eff_wave(spectrum_wave, spectrum__flux, filter_wave, filter_response, response_type='photon'):
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
    
    if response_type == 'photon':
        interp_response = np.interp(spectrum_wave, filter_wave, filter_response)
        I1 = np.trapz((spectrum_wave**2)*interp_response*spectrum__flux, spectrum_wave)
        I2 = np.trapz(spectrum_wave*interp_response*spectrum__flux, spectrum_wave)
        eff_wave = I1/I2
    
    elif response_type == 'energy':
        interp_response = np.interp(spectrum_wave, filter_wave, filter_response)
        I1 = np.trapz(spectrum_wave*interp_response*spectrum__flux, spectrum_wave)
        I2 = np.trapz(interp_response*spectrum__flux, spectrum_wave)
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
    response_type : str, default 'photon'
        Filter's response type. Either 'photon' or 'energy'. Only the Bessell filters use 'energy'.
        
    Returns
    -------
    Filter's pivot wavelength.
    
    """
    
    if response_type == 'photon':
        I1 = np.trapz(filter_response*filter_wave, filter_wave)
        I2 = np.trapz(filter_response/filter_wave, filter_wave)
        pivot_wave = np.sqrt(I1/I2)
    
    elif response_type == 'energy':
        I1 = np.trapz(filter_response, filter_wave)
        I2 = np.trapz(filter_response/(filter_wave**2), filter_wave)
        pivot_wave = np.sqrt(I1/I2)
    
    return pivot_wave


def calc_zp_ab(pivot_wave):
    """Calculates the zero point in the AB magnitude sytem given the pivot wavelength of a given filter.
    
    Parameters
    ----------
    pivot_wave : float
        Pivot wavelength.
        
    Returns
    -------
    Zero-point in the AB magnitude system.
    
    """
    
    c = 2.99792458e18  # speed of light in Angstroms
    zp_ab = 2.5*np.log10(c/pivot_wave**2) - 48.6
    
    return zp_ab


def calc_zp_vega(filter_wave, filter_response, response_type='photon'):
    """Calculates the zero point in the Vega magnitude sytem.
    
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
    Zero-point in the Vega magnitude system.
    
    """
    
    spectrum_wave, spectrum_flux = np.loadtxt('templates/alpha_lyr_stis_005.ascii').T
    f = run_filter(spectrum_wave, spectrum_flux, filter_wave, filter_response, response_type=response_type)
    zp_vega = 2.5*np.log10(f)
    
    return zp_vega


def filter_effective_range(filter_response, percent=99.9):
    """Finds the min and max indexes which contain at least the desire percentage of the filter's 
    response-function area, individually, i.e, [min:] contains the desire percentage as well as [:max]
    
    Parameters
    ----------
    filter_response : array
        Filter's response function.
    percent : float, default '99.9'
        Percentage of the filter's area that wants to be kept.
        
    Returns
    -------
    Minimum and maximum indexes which contain the wanted area of the filter, independently.
    
    """
    
    for min_index in range(len(filter_response)):
        area = 100*np.trapz(filter_response[min_index:])/np.trapz(filter_response)
        if area <= percent:
            break
            
    for max_index in reversed(range(len(filter_response))):
        area = 100*np.trapz(filter_response[:max_index])/np.trapz(filter_response)
        if area <= percent:
            break
        
    return min_index, max_index
