from filter_integration import run_filter
from gaussian_process import fit_gp
from spline import fit_spline

import numpy as np
import lmfit

def residual(params, eff_waves, flux_ratio_err, sed_wave, sed_flux, obs_flux, obs_flux_err, bands, filters, kernel, method):
    """Residual functions for the SED mangling minimization routine.
    
    Lmfit works in such a way that each parameters needs to have a residual value. In the case of the
    hyperparameters, a residual equal to the sum of the bands's residuals is used given that there is no
    model used to compare these values.
    
    Parameters
    ----------
    params : lmfit.Parameters()
        Flux values for each band to be minimized.
    eff_waves: array
        Effective wavelengths of the bands.
    flux_ratio_err : array
        "Observed" flux error values divided by the SED template values.
    sed_wave : array
        SED wavelength range
    sed_flux : array
        SED flux density values.
    obs_flux : array
        "Observed" flux values.
    obs_flux_err : array
        "Observed" flux error values.
    bands : list
        List of bands to performe minimization.
    filters : dictionary
        Dictionary with all the filters's information. Same format as 'sn.filters'.
    kernel : str
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
    method : str
        Fitting method. Either 'gp' for gaussian process or 'spline' for spline.
        
    Returns
    -------
    Array of residuals for each parameter.
    
    """
    
    param_bands = [band.lstrip("0123456789\'.-").replace("'", "").replace(".", "") for band in bands]
    flux_ratio = np.asarray([params[band].value for band in param_bands])
    min_wave, max_wave = filters[bands[0]]['wave'].min(), filters[bands[-1]]['wave'].max()
    
    if method=='gp':
        x_pred, y_pred, yerr_pred = fit_gp(eff_waves, flux_ratio, flux_ratio_err, kernel=kernel, 
                                           mangling=True, x_edges=[min_wave, max_wave])
    elif method=='spline':
        x_pred, y_pred, yerr_pred = fit_spline(eff_waves, flux_ratio, flux_ratio_err, x_edges=[min_wave, max_wave])
    
    interp_sed_flux = np.interp(x_pred, sed_wave, sed_flux)
    mangled_wave, mangled_flux = x_pred, y_pred*interp_sed_flux
    model_flux = np.array([run_filter(mangled_wave, mangled_flux, filters[band]['wave'], filters[band]['transmission'], 
                                      filters[band]['response_type']) for band in bands])

    residuals = np.abs((obs_flux - model_flux)/obs_flux_err)

    return residuals


def mangle(flux_ratio, flux_ratio_err, sed_wave, sed_flux, bands, filters, obs_fluxes, obs_flux_err, kernel='squaredexp', method='gp'):
    """Mangling routine.
    
    A mangling of the SED is done by minimizing the the difference between the "observed" fluxes and the fluxes
    coming from the modified SED.
    
    Parameters
    ----------
    flux_ratio : array
        "Observed" flux values divided by the SED template values.
    flux_ratio_err : array
        "Observed" flux error values divided by the SED template values.
    sed_wave : array
        SED wavelength range
    sed_flux : array
        SED flux density values.
    bands : list
        List of bands to performe minimization.
    filters : dictionary
        Dictionary with all the filters's information. Same format as 'sn.filters'.
    obs_flux : array
        "Observed" flux values.
    obs_flux_err : array
        "Observed" flux error values.
    kernel : str
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
    method : str
        Fitting method. Either 'gp' for gaussian process or 'spline' for spline.
        
    Returns
    -------
    Returns the mangled/modified SED with 1-sigma standard deviation and all the results
    from the mangling routine (these can plotted later to check the results).
    
    """

    #### optimize values ####
    eff_waves = np.asarray([filters[band]['eff_wave'] for band in bands])
    
    params = lmfit.Parameters()
    # lmfit Parameters doesn't allow parameter names beginning with numbers all digits are deleted 
    # just in case and quotes (') and dots (.) as well.
    param_bands = [band.lstrip("0123456789\'.-").replace("'", "").replace(".", "") for band in bands]  
    
    for val, band in zip(flux_ratio, param_bands):                
        params.add(band, value=val, min=val*0)#.99, max=val*1.01)   # tighten this constrains for a smoother mangling
         
    args=(eff_waves, flux_ratio_err, sed_wave, sed_flux, obs_fluxes, obs_flux_err, bands, filters, kernel, method)
    result = lmfit.minimizer.minimize(fcn=residual, params=params, args=args)
    
    #### use optimized results ####
    opt_flux_ratio = np.asarray([result.params[band].value for band in param_bands])
    min_wave, max_wave = filters[bands[0]]['wave'].min(), filters[bands[-1]]['wave'].max()
    
    if method=='gp':
        x_pred, y_pred, yerr_pred = fit_gp(eff_waves, opt_flux_ratio, flux_ratio_err, kernel=kernel,
                                           mangling=True, x_edges=[min_wave, max_wave])
    elif method=='spline':
        kernel = None  # for result display purposes
        x_pred, y_pred, yerr_pred = fit_spline(eff_waves, opt_flux_ratio, flux_ratio_err, x_edges=[min_wave, max_wave])
        
    interp_sed_flux = np.interp(x_pred, sed_wave, sed_flux)
    mangled_wave, mangled_flux, mangled_flux_err = x_pred, y_pred*interp_sed_flux, yerr_pred*interp_sed_flux
    
    #### propagate errors ####
    flux_diffs = []
    for band, obs_flux in zip(bands, obs_fluxes):
        model_flux = run_filter(mangled_wave, mangled_flux, 
                                filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type'])
        flux_diffs.append(obs_flux - model_flux)
    flux_diffs = np.r_[0, flux_diffs, 0]    
    
    interp_flux_err = np.interp(mangled_wave, np.r_[min_wave, eff_waves, max_wave], flux_diffs)
    mangled_flux_err = np.sqrt(mangled_flux_err**2 + interp_flux_err**2)
    
    yerr_pred = mangled_flux_err/interp_sed_flux  # error propagation from flux_diffs
    
    #### save results ####
    sed_fluxes = np.empty(0)
    for band in bands:
        band_flux = run_filter(sed_wave, sed_flux, 
                               filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type'])
        sed_fluxes = np.r_[sed_fluxes, band_flux]
    
    mangling_results = {'init_vals':{'waves':eff_waves, 'flux_ratios':flux_ratio, 'flux_ratios_err':flux_ratio_err},
                        'opt_vals':{'waves':eff_waves, 'flux_ratios':opt_flux_ratio, 'flux_ratios_err':flux_ratio_err},
                        'sed_vals':{'waves':eff_waves, 'fluxes':sed_fluxes},
                        'obs_vals':{'waves':eff_waves, 'fluxes':obs_fluxes, 'fluxes_err':obs_flux_err},
                        'opt_fit':{'waves':x_pred, 'flux_ratios':y_pred, 'flux_ratios_err':yerr_pred},
                        'init_sed':{'wave':sed_wave, 'flux':sed_flux},
                        'mangled_sed':{'wave':mangled_wave, 'flux':mangled_flux, 'flux_err':mangled_flux_err},
                        'kernel':kernel, 
                        'method':method, 
                        'result':result}
    
    return mangled_wave, mangled_flux, mangled_flux_err, mangling_results
