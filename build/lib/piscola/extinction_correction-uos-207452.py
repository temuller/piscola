from filter_integration import run_filter
import extinction
import sfdmap

import matplotlib.pyplot as plt
import numpy as np

def redden(wave, flux, ra, dec, scaling=0.86):
    """Reddens the given spectrum, given a right ascension and declination.
        
    Parameters
    ----------
    wave : array
        Wavelength values.
    flux : array
        Flux density values.
    ra : float
        Right ascension.
    dec : float
        Declination
    scaling: float, default '0.86'
        Recalibration of the Schlegel, Finkbeiner & Davis (1998) dust map. Either '0.86'
        for the Schlafly & Finkbeiner (2011) recalibration or '1.0' for the original 
        dust map of Schlegel, Finkbeiner & Davis (1998).
        
    Returns
    -------
    Returns the redden flux density values.
    
    """
    
    m = sfdmap.SFDMap(mapdir='./sfddata-master/', scaling=scaling) 
    ebv = m.ebv(ra, dec) # RA and DEC in degrees
    r_v  = 3.1
    a_v = r_v*ebv
    ext = extinction.ccm89(wave, a_v, r_v)
    redden_flux = extinction.apply(ext, flux) # applies extinction to flux
    
    return redden_flux


def deredden(wave, flux, ra, dec, scaling=0.86):
    """Dereddens the given spectrum, given a right ascension and declination.
        
    Parameters
    ----------
    wave : array
        Wavelength values.
    flux : array
        Flux density values.
    ra : float
        Right ascension.
    dec : float
        Declination
    scaling: float, default '0.86'
        Recalibration of the Schlegel, Finkbeiner & Davis (1998) dust map. Either '0.86'
        for the Schlafly & Finkbeiner (2011) recalibration or '1.0' for the original 
        dust map of Schlegel, Finkbeiner & Davis (1998).
        
    Returns
    -------
    Returns the deredden flux density values.
    
    """
    
    m = sfdmap.SFDMap(mapdir='./sfddata-master/', scaling=scaling) 
    ebv = m.ebv(ra, dec) # RA and DEC in degrees
    r_v  = 3.1
    a_v = r_v*ebv
    ext = extinction.ccm89(wave, a_v, r_v)
    deredden_flux = extinction.apply(-ext, flux) #removes extinction from flux
    
    return deredden_flux


def extinction_filter(filter_wave, filter_response, ra, dec, scaling=0.86):
    """Estimate the extinction for a given filter, given a right ascension and declination.
        
    Parameters
    ----------
    filter_wave : array
        Filter's wavelength range.
    filter_response : array
        Filter's response function.
    ra : float
        Right ascension.
    dec : float
        Declination
    scaling: float, default '0.86'
        Recalibration of the Schlegel, Finkbeiner & Davis (1998) dust map. Either '0.86'
        for the Schlafly & Finkbeiner (2011) recalibration or '1.0' for the original 
        dust map of Schlegel, Finkbeiner & Davis (1998).
        
    Returns
    -------
    Returns the extinction value in magnitude.
    
    """
    
    flux = 100
    deredden_flux = deredden(filter_wave, flux, ra, dec, scaling=scaling)

    f1 = run_filter(filter_wave, flux, filter_wave, filter_response)
    f2 = run_filter(filter_wave, deredden_flux, filter_wave, filter_response)
    A = -2.5*np.log10(f1/f2)
    
    return A


def extinction_curve(ra, dec, scaling=0.86):
    """Plots the extinction curve for a given RA and DEC.
    
    Parameters
    ----------
    ra : float
        Right ascension.
    dec : float
        Declination
    scaling: float, default '0.86'
        Recalibration of the Schlegel, Finkbeiner & Davis (1998) dust map. Either '0.86'
        for the Schlafly & Finkbeiner (2011) recalibration or '1.0' for the original 
        dust map of Schlegel, Finkbeiner & Davis (1998).
    
    """
    
    flux = 100
    wave = np.arange(1000, 25001)  # in Angstroms
    deredden_flux = deredden(wave, flux, ra, dec, scaling=scaling)
    ff = 1 - flux/deredden_flux
    
    f, ax = plt.subplots(figsize=(8,6))
    ax.plot(wave, ff)
    
    ax.set_xlabel(r'wavelength ($\AA$)', fontsize=18)
    ax.set_ylabel('fraction of extinction', fontsize=18)
    ax.set_title(r'Extinction curve', fontsize=18)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    
    plt.show()
