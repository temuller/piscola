from .filter_utils import integrate_filter
from .gaussian_process import gp_mf_fit, spline_mf_fit

import numpy as np
import lmfit

def residual(params, flux_ratios_err_array, wave_array, sed_wave, sed_flux, obs_flux, norm, bands,
                                                                    filters, method, kernel, x_edges, linear_extrap):
    """Residual function for the mangling minimization routine.

    Parameters
    ----------
    params : lmfit.Parameters()
        Flux values for each band to be minimized.
    flux_ratios_err_array : array
        "Observed" flux errors divided by the SED template flux values.
    wave_array: array
        Effective wavelengths of the bands.
    sed_wave : array
        SED wavelength range
    sed_flux : array
        SED flux density values.
    obs_flux : array
        "Observed" flux values.
    norm : float
        Normalization value to improve minimization.
    bands : list
        List of bands to performe minimization.
    filters : dictionary
        Dictionary with all the filters's information. Same format as :func:`sn.filters`.
    method : str
        Method to estimate the mangling function. Either ``gp`` or ``spline``.
    kernel : str
        Kernel to be used with the gaussian process. Possible choices are: ``matern52``, ``matern32``, ``squaredexp``.
    x_edges : array-like
        Minimum and maximum x-axis values. These are used to extrapolate both edges.
    linear_extrap: bool
        Type of extrapolation for the edges. Linear if ``True``, free if ``False``.

    Returns
    -------
    residuals : array
    Array of residuals for each parameter.

    """

    param_bands = [band.lstrip("0123456789\'.-").replace("'", "").replace(".", "") for band in bands]
    flux_ratio_array = np.array([params[band].value for band in param_bands])

    if method=='gp':
        x_pred, y_pred, yerr_pred, _ = gp_mf_fit(wave_array, flux_ratio_array, yerr_data=0.0, kernel=kernel,
                                                x_edges=x_edges, linear_extrap=linear_extrap)
    elif method=='spline':
        x_pred, y_pred, yerr_pred, _ = spline_mf_fit(wave_array, flux_ratio_array, yerr_data=0.0,
                                                x_edges=x_edges, linear_extrap=linear_extrap)

    interp_sed_flux = np.interp(x_pred, sed_wave, sed_flux, left=0.0, right=0.0)
    mangled_wave, mangled_flux = x_pred, (y_pred*norm)*interp_sed_flux
    model_flux = np.array([integrate_filter(mangled_wave, mangled_flux, filters[band]['wave'], filters[band]['transmission'],
                                      filters[band]['response_type']) for band in bands])

    residuals = np.abs(2.5*np.log10(obs_flux/model_flux))
    residuals = np.nan_to_num(residuals, nan=np.nanmean(residuals))  # replace nan with mean value, if any

    return residuals


def mangle(wave_array, flux_ratio_array, flux_ratios_err_array, sed_wave, sed_flux, obs_fluxes, obs_errs,
                                                                bands, filters, method, kernel, x_edges, linear_extrap):
    """Estimates the mangling function. This is done by minimizing the difference between the observed fluxes and the fluxes
    from the SED template.

    Parameters
    ----------
    wave_array: array
        Effective wavelengths of the bands.
    flux_ratio_array : array
        "Observed" flux values divided by the SED template flux values.
    flux_ratios_err_array : array
        "Observed" flux errors divided by the SED template flux values.
    sed_wave : array
        SED wavelength range
    sed_flux : array
        SED flux density values.
    obs_fluxes : array
        "Observed" flux values.
    obs_errs : array
        "Observed" flux error values.
    bands : list
        List of bands to performe minimization.
    filters : dictionary
        Dictionary with all the filters's information. Same format as :func:`sn.filters`.
    method : str
        Method to estimate the mangling function. Either ``gp`` or ``spline``.
    kernel : str
        Kernel to be used with the gaussian process. Possible choices are: ``matern52``, ``matern32``, ``squaredexp``.
    x_edges : array-like
        Minimum and maximum x-axis values. These are used to extrapolate both edges.
    linear_extrap: bool
        Type of extrapolation for the edges. Linear if ``True``, free if ``False``.

    Returns
    -------
    mangling_results : dict
        Dictionary with the mangled SED, standard deviation (:math:`1\sigma`) and all the results
        from the mangling routine (these can be plotted later to check the results).

    """

    #########################
    #### Optimise values ####
    #########################
    params = lmfit.Parameters()
    # lmfit Parameters doesn't allow parameter names beginning with numbers, so all digits are deleted.
    # Quotes (') and dots (.) are always removed from the names.
    param_bands = [band.lstrip("0123456789\'.-").replace("'", "").replace(".", "") for band in bands]

    norm = flux_ratio_array.max()  # normalization avoids tiny numbers which cause problems with the minimization routine
    for val, band in zip(flux_ratio_array/norm, param_bands):
        params.add(band, value=val, min=0) # , max=val*1.2)   # tighten this constrains for a smoother(?) mangling

    args=(flux_ratios_err_array, wave_array, sed_wave, sed_flux, obs_fluxes, norm,
                                                                bands, filters, method, kernel, x_edges, linear_extrap)
    lmfit_results = lmfit.minimizer.minimize(fcn=residual, params=params, args=args, xtol=1e-4, ftol=1e-4, max_nfev=80)

    ###############################
    #### Use Optimized Results ####
    ###############################
    if lmfit_results.success:
        opt_flux_ratio = np.array([lmfit_results.params[band].value for band in param_bands]) * norm
    else:
        opt_flux_ratio = np.array([lmfit_results.params[band].init_value for band in param_bands]) * norm

    if method=='gp':
        x_pred, y_pred, yerr_pred, fitting_results = gp_mf_fit(wave_array, opt_flux_ratio, yerr_data=0.0, kernel=kernel,
                                                x_edges=x_edges, linear_extrap=linear_extrap)
    elif method=='spline':
        x_pred, y_pred, yerr_pred, fitting_results = spline_mf_fit(wave_array, opt_flux_ratio, yerr_data=0.0,
                                                x_edges=x_edges, linear_extrap=linear_extrap)

    interp_sed_flux = np.interp(x_pred, sed_wave, sed_flux, left=0.0, right=0.0)
    mangled_wave, mangled_flux, mangled_flux_err = x_pred, y_pred*interp_sed_flux, yerr_pred*interp_sed_flux

    ###########################
    #### Error propagation ####
    ###########################
    extended_wave_array = np.r_[x_edges[0], wave_array, x_edges[-1]]

    flux_diffs = []
    flux_diff_ratios = []
    for band, obs_flux in zip(bands, obs_fluxes):
        model_flux = integrate_filter(mangled_wave, mangled_flux,
                                filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type'])
        flux_diffs.append(obs_flux - model_flux)
        flux_diff_ratios.append(obs_flux/model_flux)
    flux_diff_ratios =np.array(flux_diff_ratios)

    # penalise for inaccuracy in optimisation routine of the mangling function
    extended_flux_diffs = np.r_[flux_diffs[0], flux_diffs, flux_diffs[-1]]
    interp_mangling_err = np.interp(mangled_wave, extended_wave_array, extended_flux_diffs, left=0.0, right=0.0)
    mangled_flux_err = np.sqrt(mangled_flux_err**2 + interp_mangling_err**2)

    # add uncertainties from the observed (gp-fit) fluxes for each band
    error_prop = np.r_[obs_errs[0], obs_errs, obs_errs[-1]]  # edges extrapolaton of the uncertainties
    interp_error_prop = np.interp(mangled_wave, extended_wave_array, error_prop, left=0.0, right=0.0)
    mangled_flux_err = np.sqrt(mangled_flux_err**2 + interp_error_prop**2)

    # Save Results
    sed_fluxes = []
    for band in bands:
        sed_fluxes.append(integrate_filter(sed_wave, sed_flux,
                               filters[band]['wave'], filters[band]['transmission'], filters[band]['response_type']))
    sed_fluxes = np.array(sed_fluxes)

    mangling_results = {'init_flux_ratios':{'waves':wave_array, 'flux_ratios':flux_ratio_array},
                        'opt_flux_ratios':{'waves':wave_array, 'flux_ratios':opt_flux_ratio},
                        'flux_ratios':flux_diff_ratios,
                        'sed_band_fluxes':{'waves':wave_array, 'fluxes':sed_fluxes},
                        'obs_band_fluxes':{'waves':wave_array, 'fluxes':obs_fluxes},
                        'mangling_function':{'waves':x_pred, 'flux_ratios':y_pred, 'flux_ratios_err':yerr_pred},
                        'init_sed':{'wave':sed_wave, 'flux':sed_flux},
                        'mangled_sed':{'wave':mangled_wave, 'flux':mangled_flux, 'flux_err':mangled_flux_err},
                        'fitting_results':fitting_results,
                        'kernel':kernel,
                        'lmfit_results':lmfit_results}

    return mangling_results
