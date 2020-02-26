from .filter_integration import run_filter
from .gaussian_process import fit_gp
from .spline import fit_spline

import numpy as np
import lmfit

def residual(params, wave_array, sed_wave, sed_flux, obs_flux, norm, bands, filters, kernel, x_edges):
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
    flux_ratio_array = np.array([params[band].value for band in param_bands])

    x_pred, y_pred, yerr_pred = fit_gp(wave_array, flux_ratio_array, np.zeros_like(flux_ratio_array),
                                        kernel=kernel, x_edges=x_edges)

    interp_sed_flux = np.interp(x_pred, sed_wave, sed_flux)
    mangled_wave, mangled_flux = x_pred, (y_pred*norm)*interp_sed_flux
    model_flux = np.array([run_filter(mangled_wave, mangled_flux, filters[band]['wave'], filters[band]['transmission'],
                                      filters[band]['response_type']) for band in bands])

    residuals = -2.5*np.log10(obs_flux/model_flux)
    return residuals


def mangle(wave_array, flux_ratio_array, sed_wave, sed_flux, bands, filters, obs_fluxes, kernel, x_edges):
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

    #########################
    #### Optimise values ####
    #########################
    params = lmfit.Parameters()
    # lmfit Parameters doesn't allow parameter names beginning with numbers so all digits are deleted
    # just in case and quotes (') and dots (.) as well.
    param_bands = [band.lstrip("0123456789\'.-").replace("'", "").replace(".", "") for band in bands]

    norm = flux_ratio_array.max()  # normalization avoids tiny numbers which cause problems with the minimization routine
    for val, band in zip(flux_ratio_array/norm, param_bands):
        params.add(band, value=val, min=0) # , max=val*1.2)   # tighten this constrains for a smoother(?) mangling

    args=(wave_array, sed_wave, sed_flux, obs_fluxes, norm, bands, filters, kernel, x_edges)
    result = lmfit.minimizer.minimize(fcn=residual, params=params, args=args, xtol=1e-3, ftol=1e-3, maxfev=40)

    ###############################
    #### Use Optimized Results ####
    ###############################
    opt_flux_ratio = np.asarray([result.params[band].value for band in param_bands]) * norm

    x_pred, y_pred, yerr_pred = fit_gp(wave_array, opt_flux_ratio, np.zeros_like(opt_flux_ratio),
                                        kernel=kernel, x_edges=x_edges)

    interp_sed_flux = np.interp(x_pred, sed_wave, sed_flux)
    mangled_wave, mangled_flux, mangled_flux_err = x_pred, y_pred*interp_sed_flux, yerr_pred*interp_sed_flux

    ###########################
    #### Error propagation ####
    ###########################
    flux_diffs = []
    for band, obs_flux in zip(bands, obs_fluxes):
        model_flux = run_filter(mangled_wave, mangled_flux,
                                filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type'])
        flux_diffs.append(obs_flux - model_flux)
    flux_diffs = np.r_[flux_diffs[0], flux_diffs, flux_diffs[-1]]

    # penalise for inaccuracy in optimisation routine
    interp_flux_err = np.interp(mangled_wave, np.r_[x_edges[0], wave_array, x_edges[-1]], flux_diffs)
    mangled_flux_err = np.sqrt(mangled_flux_err**2 + interp_flux_err**2)

    yerr_pred = mangled_flux_err/interp_sed_flux  # error propagation from flux_diffs

    ######################
    #### Save Results ####
    ######################
    sed_fluxes = np.empty(0)
    for band in bands:
        band_flux = run_filter(sed_wave, sed_flux,
                               filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type'])
        sed_fluxes = np.r_[sed_fluxes, band_flux]

    mangling_results = {'init_vals':{'waves':wave_array, 'flux_ratios':flux_ratio_array, 'flux_ratios_err':0.0},
                        'opt_vals':{'waves':wave_array, 'flux_ratios':opt_flux_ratio, 'flux_ratios_err':0.0},
                        'sed_vals':{'waves':wave_array, 'fluxes':sed_fluxes},
                        'obs_vals':{'waves':wave_array, 'fluxes':obs_fluxes},
                        'opt_fit':{'waves':x_pred, 'flux_ratios':y_pred, 'flux_ratios_err':yerr_pred},
                        'init_sed':{'wave':sed_wave, 'flux':sed_flux},
                        'mangled_sed':{'wave':mangled_wave, 'flux':mangled_flux, 'flux_err':mangled_flux_err},
                        'kernel':kernel,
                        'result':result}

    #return mangled_wave, mangled_flux, mangled_flux_err, mangling_results
    return mangling_results
