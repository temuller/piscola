import jax
import jaxopt
import numpy as np
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels, transforms

jax.config.update("jax_enable_x64", True)

def prepare_gp_inputs(times, wavelengths, fluxes, flux_errors, fit_type, wave_log):
    """Prepares the inputs for the Gaussian Process model fitting.
    
    Parameters
    ----------
    times: ndarray 
        Light-curve epochs.
    wavelengths: ndarray 
        Light-curve effective wavelength.
    fluxes: ndarray 
        Light-curve fluxes.
    flux_errors: ndarray 
        Light-curve flux errors.
    fit_type: str
        Transformation used for the light-curve fits: ``flux``, ``log``, ``arcsinh``.
    wave_log: bool
        Whether to use logarithmic (base 10) scale for the 
        wavelength axis.
        
    Returns
    -------
    X: ndarray
        X-axis array for the Gaussian Process model.
    y: ndarray
        Y-axis array for the Gaussian Process model.
    yerr: ndarray
        Y-axis errors for the Gaussian Process model.
    y_norm: float
        Normalisation used for the fluxes and errors. The maximum
        of the fluxes is used.
    """
    valid_fit_types = ["flux", "log", "arcsinh"]
    err_message = f"Not a valid fit type ({valid_fit_types}): {fit_type}"
    assert fit_type in valid_fit_types, err_message

    X = (times, wavelengths)
    if wave_log is True:
        X = (times, jnp.log10(wavelengths) * 10)

    # normalise fluxes - values have to be ideally above zero
    y_norm = np.copy(fluxes.max()) 
    if fit_type == "flux":
        y = (fluxes / y_norm).copy() 
        yerr = (flux_errors / y_norm).copy()
    elif fit_type == "log":
        mask = fluxes > 0.0
        #y_norm = np.copy(fluxes[mask].mean())
        if len(fluxes) == len(X[0]):
            # when predicting, the arrays do not match in length
            X = (X[0][mask], X[1][mask])
        y = np.log10(fluxes[mask] / y_norm)
        yerr = np.abs(flux_errors / (fluxes * np.log(10)))[mask]
    elif fit_type == "arcsinh":
        if len(fluxes) == len(X[0]):
            # when predicting, the arrays do not match in length
            X = (X[0], X[1])
        y = np.arcsinh(fluxes / y_norm) 
        yerr = (flux_errors / y_norm) / np.sqrt((fluxes / y_norm) ** 2 + 1)

    return X, y, yerr, y_norm

def fit_gp_model(times, wavelengths, fluxes, flux_errors, k1='Matern52', fit_mean=False, 
                 fit_type='flux', wave_log=True, time_scale=None, wave_scale=None, add_noise=True):
    """Fits a Gaussian Process model to a SN multi-colour light curve.
    
    All input arrays MUST have the same length.

    Parameters
    ----------
    times: ndarray 
        Light-curve epochs.
    wavelengths: ndarray 
        Light-curve effective wavelength.
    fluxes: ndarray 
        Light-curve fluxes.
    flux_errors: ndarray 
        Light-curve flux errors.
    k1: str
        Kernel to be used for the time axis. Either ``Matern52``,
        ``Matern32`` or ``ExpSquared``.
    fit_type: str
        Transformation used for the light-curve fits: ``flux``, ``log``, ``arcsinh``.
    wave_log: bool, default ``True``.
        Whether to use logarithmic (base 10) scale for the 
        wavelength axis.
    time_scale: float, default ``None``
        If given, the time scale is fixed using this value, in units of days.
    wave_scale: float, default ``None``
        If given, the wavelength scale is fixed using this value, in units of angstroms.
        Note that if 'wave_log=True', the logarithm base 10 of this value is used.
    add_noise: bool, default ``True``
        Whether to add a white-noise component to the GP model. This "inflates" the errors.
        
    Returns
    -------
    gp_model: ~tinygp.gp.GaussianProcess
        Gaussian Process multi-colour light-curve model.
    """
    assert k1 in ['Matern52', 'Matern32', 'ExpSquared'], "Not a valid kernel"
    def build_gp(params):
        """Creates a Gaussian Process model.
        """
        nonlocal time_scale, wave_scale, add_noise  # import from main function
        if time_scale is None:
            log_time_scale = params["log_scale"][0]
        else:
            log_time_scale = np.log(time_scale)
        if wave_scale is None:
            log_wave_scale = params["log_scale"][-1]
        else:
            log_wave_scale = np.log(wave_scale)
        if add_noise is True:
            noise = jnp.exp(2 * params["log_noise"])
        else:
            noise = 0.0

        # select time-axis kernel
        if k1 == 'Matern52':
            kernel1 = transforms.Subspace(0, kernels.Matern52(scale=jnp.exp(log_time_scale)))
        elif k1 == 'Matern32':
            kernel1 = transforms.Subspace(0, kernels.Matern32(scale=jnp.exp(log_time_scale)))
        else:
            kernel1 = transforms.Subspace(0, kernels.ExpSquared(scale=jnp.exp(log_time_scale)))
        # wavelength-axis kernel
        kernel2 = transforms.Subspace(1, kernels.ExpSquared(scale=jnp.exp(log_wave_scale)))
        
        kernel = jnp.exp(params["log_amp"]) * kernel1 * kernel2
        diag = yerr ** 2 + noise
        #ids = np.hstack([np.zeros_like(X[1][X[1] == wave]).astype(int) + i for i, wave in enumerate(np.unique(X[1]))])
        #diag = yerr ** 2 + jnp.exp(2 * params["log_jitter"][ids])
        
        if fit_mean is True:
            mean = jnp.exp(params["log_mean"])
        elif fit_type == "log":
            mean = -jnp.exp(params["log_mean"])
        else:
            mean = None

        return GaussianProcess(kernel, X, diag=diag, mean=mean)

    @jax.jit
    def loss(params):
        """Loss function for the Gaussian Process hyper-parameters optimisation.
        """
        return -build_gp(params).condition(y).log_probability
    
    X, y, yerr, _ = prepare_gp_inputs(times, wavelengths, fluxes, 
                                            flux_errors, fit_type=fit_type, 
                                            wave_log=wave_log)

    # GP hyper-parameters
    scales = np.array([30, 2000]) # units: days, angstroms
    if wave_log is True:
        # the approx range from opt to NIR is 1 in log space
        scales = np.array([30, 0.5]) # units: days, log10(angstroms)
    
    params = {
        "log_amp": jnp.log(y.var()),
        "log_scale": jnp.log(scales),
        "log_noise": jnp.log(np.mean(yerr)),
        #"log_jitter": np.zeros_like(np.unique(X[1])),
    }
    if fit_mean is True:
        params.update({"log_mean": jnp.log(np.average(y, weights=1/yerr**2))})
    elif fit_type == "log":
        # absolute value to avoid negative in log
        params.update({"log_mean": jnp.log(np.abs(y.min()))})

    # Train the GP model
    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params)
    gp_model = build_gp(soln.params)
    
    return gp_model

def fit_single_lightcurve(times, fluxes, flux_errors, k1='Matern52'):
    """Fits a Gaussian Process model to a single SN light curve.
    
    All input arrays MUST have the same length.

    Parameters
    ----------
    times: ndarray 
        Light-curve epochs.
    fluxes: ndarray 
        Light-curve fluxes.
    flux_errors: ndarray 
        Light-curve flux errors.
    k1: str
        Kernel to be used for the time axis. Either ``Matern52``,
        ``Matern32`` or ``ExpSquared``.
        
    Returns
    -------
    gp_model: ~tinygp.gp.GaussianProcess
        Gaussian Process light-curve model.
    """
    assert k1 in ['Matern52', 'Matern32', 'ExpSquared'], "Not a valid kernel"
    def build_gp(params):
        """Creates a Gaussian Process model.
        """
        # select time-axis kernel
        if k1 == 'Matern52':
            kernel1 = kernels.Matern52(scale=jnp.exp(params["log_scale"]))
        elif k1 == 'Matern32':
            kernel1 = kernels.Matern32(scale=jnp.exp(params["log_scale"]))
        else:
            kernel1 = kernels.ExpSquared(scale=jnp.exp(params["log_scale"]))
        kernel = jnp.exp(params["log_amp"]) * kernel1
        diag = yerr ** 2 

        return GaussianProcess(kernel, x, diag=diag)

    @jax.jit
    def loss(params):
        """Loss function for the Gaussian Process hyper-parameters optimisation.
        """
        return -build_gp(params).condition(y).log_probability
    
    x = times.copy()
    y = fluxes.copy() / fluxes.max()
    yerr = flux_errors.copy() / fluxes.max()
    
    # GP hyper-parameters
    time_scale = 10  # days
    params = {
        "log_amp": jnp.log(y.var()),
        "log_scale": jnp.log(time_scale),
    }
    # Train the GP model
    solver = jaxopt.ScipyMinimize(fun=loss)
    soln = solver.run(params)
    gp_model = build_gp(soln.params)
    
    return gp_model