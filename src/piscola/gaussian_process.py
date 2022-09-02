import scipy
import numpy as np
from functools import partial

import george
from george.kernels import Matern52Kernel, Matern32Kernel, ExpSquaredKernel


def gp_lc_fit(x_data, y_data, yerr_data=0.0, kernel="matern52"):
    r"""Fits a single light curve with Gaussian Process.

    The package ``george`` is used for the gaussian process fit.

    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, default ``0.0``
        Dependent value errors.
    kernel : str, default ``matern52``
        Kernel to be used with the gaussian process. E.g., ``matern52``, ``matern32``, ``squaredexp``.

    Returns
    -------
    x_pred : array
        Interpolated x-axis values.
    mean : array
        Interpolated  values.
    std : array
        Standard deviation (:math:`1\sigma`) of the interpolation.

    """

    # define the objective function (negative log-likelihood in this case)
    def neg_log_like(params):
        """Negative log-likelihood."""
        gp.set_parameter_vector(params)
        log_like = gp.log_likelihood(y, quiet=True)
        if np.isfinite(log_like):
            return -log_like
        else:
            return np.inf

    # and the gradient of the objective function
    def grad_neg_log_like(params):
        """Gradient of the negative log-likelihood."""
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y, quiet=True)

    x, y, yerr = np.copy(x_data), np.copy(y_data), np.copy(yerr_data)
    x_min, x_max = x.min(), x.max()

    y_norm = y.max()
    y /= y_norm
    yerr /= y_norm

    var, length_scale = np.var(y), np.diff(x).max()

    if kernel == "matern52":
        ker = var * Matern52Kernel(length_scale**2)
    elif kernel == "matern32":
        ker = var * Matern32Kernel(length_scale**2)
    elif kernel == "squaredexp":
        ker = var * ExpSquaredKernel(length_scale**2)
    else:
        raise ValueError(f'"{kernel}" is not a valid kernel.')

    gp = george.GP(kernel=ker, fit_mean=True)
    # initial guess
    gp.compute(x, yerr)

    # optimization routine for hyperparameters
    p0 = gp.get_parameter_vector()
    results = scipy.optimize.minimize(
        neg_log_like, p0, jac=grad_neg_log_like, method="L-BFGS-B"
    )
    gp.set_parameter_vector(results.x)

    step = 0.05  # days
    x_pred = np.arange(x_min, x_max + step, step)

    mean, var = gp.predict(y, x_pred, return_var=True)
    std = np.sqrt(var)

    x_pred, y_pred, yerr_pred = x_pred, mean * y_norm, std * y_norm

    return x_pred, y_pred, yerr_pred


def gp_2d_fit(
    x1_data,
    x2_data,
    y_data,
    yerr_data=0.0,
    kernel1="matern52",
    kernel2="squaredexp",
    gp_mean="mean",
    x1_ext=(5, 10),
    x2_ext=(1000, 2000),
):
    r"""Fits multi-colour light curves in 2D with Gaussian Process.

    **Note1:** ``x1`` refers to `time` axis while ``x2`` refers to `wavelength` axis.

    **Note2:** available kernels are: ``matern52``, ``matern32`` and
    ``squaredexp``.

    Parameters
    ----------
    x1_data: array
        Time axis data.
    x2_data: array
        Wavelength axis data.
    y_data: array
        Any data: fluxes, magnitudes, flux ratios, etc.
    yerr_data: array or float, default ``0.0``
        Errors on y_data.
    kernel1: str, default ``matern52``
        Kernel for the time axis.
    kernel2: str, default ``squaredexp``
        Kernel for the wavelength acis.
    gp_mean: str, default ``mean``
        Gaussian process mean function. Either ``mean``, ``max`` or ``min``.
    x1_ext: str, default ``(5, 10)``
        Extrapolation "leftward" and "rightward" for the time axis.
    x2_ext: str, default ``(1000, 2000)``
        Extrapolation "leftward" and "rightward" for the wavelength axis.

    Returns
    -------
    X_predict: array
        Interpolated 2D x-axis grid.
    y_pred: array
        Interpolated  values.
    yerr_pred: array
        Standard deviation (:math:`1\sigma`) of the interpolation.
    gp: ~george.gp
        Gaussian Process model.
    """

    # define the objective function for optimization
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    # and the gradient of the objective function
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    x1, x2 = np.copy(x1_data), np.copy(x2_data)
    y, yerr = np.copy(y_data), np.copy(yerr_data)

    # normalize data
    y_norm = y.max()
    y /= y_norm
    yerr /= y_norm
    # reshape x-axis for george
    X = np.array([x1, x2]).reshape(2, -1).T

    # GP kernels
    kernels_dict = {
        "matern52": Matern52Kernel,
        "matern32": Matern32Kernel,
        "squaredexp": ExpSquaredKernel,
    }

    valid_kernels = list(kernels_dict.keys())
    err_message = f"Invalid kernel. Choose between:{valid_kernels}"
    assert kernel1 in valid_kernels, err_message
    assert kernel2 in valid_kernels, err_message

    # GP hyperparameters
    var = np.var(y)
    length1 = np.diff(x1).max()
    length2 = np.diff(x2).max()

    ker1 = kernels_dict[kernel1](length1**2, ndim=2, axes=0)
    ker2 = kernels_dict[kernel2](length2**2, ndim=2, axes=1)
    ker = var * ker1 * ker2

    # GP mean function
    mean_dict = {"mean": y.mean(), "min": y.min(), "max": y.max()}
    mean_func = mean_dict[gp_mean]

    gp = george.GP(kernel=ker, mean=mean_func, fit_mean=True)
    # initial guess
    gp.compute(X, yerr)

    # optimization routine for hyperparameters
    p0 = gp.get_parameter_vector()
    try:
        results = scipy.optimize.minimize(
            neg_ln_like, p0, jac=grad_neg_ln_like, method="BFGS"
        )
    except:
        results = scipy.optimize.minimize(
            neg_ln_like, p0, jac=grad_neg_ln_like, method="L-BFGS-B"
        )
    gp.set_parameter_vector(results.x)

    # extrapolation edges
    x1_min, x1_max = x1.min() - x1_ext[0], x1.max() + x1_ext[1]
    x2_min, x2_max = x2.min() - x2_ext[0], x2.max() + x2_ext[1]

    # x-axis prediction array
    step1 = 0.1  # in days
    step2 = 10  # in angstroms
    x1_pred = np.arange(x1_min, x1_max + step1, step1)
    x2_pred = np.arange(x2_min, x2_max + step2, step2)
    X_predict = np.array(np.meshgrid(x1_pred, x2_pred))
    X_predict = X_predict.reshape(2, -1).T

    mean, var = gp.predict(y, X_predict, return_var=True)
    std = np.sqrt(var)
    # de-normalize results
    y_pred = mean * y_norm
    yerr_pred = std * y_norm

    return X_predict, y_pred, yerr_pred, gp
