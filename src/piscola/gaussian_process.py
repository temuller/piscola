import numpy as np
import george
import scipy

from .pisco_utils import extrapolate_mangling_edges
from scipy.interpolate import UnivariateSpline
from functools import partial

def gp_lc_fit(x_data, y_data, yerr_data=0.0, kernel='matern52'):
    """Fits a single light curve with gaussian process.

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
    x_pred*x_norm : array
        Interpolated x-axis values.
    mean*y_norm : array
        Interpolated  values.
    std*y_norm : array
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

    # normalise the data for better results
    x_norm = 1e4
    x /= x_norm
    x_min, x_max = x.min(), x.max()

    y_norm = y.max()
    y /= y_norm
    yerr /= y_norm

    var, length_scale = np.var(y), np.diff(x).max()
    bounds_var, bounds_length = [(np.log(1e-6), np.log(10))], [(np.log(1e-8), np.log(1e2))]

    k1 = george.kernels.ConstantKernel(np.log(var))

    if kernel == 'matern52':
        k2 = george.kernels.Matern52Kernel(length_scale**2)
    elif kernel == 'matern32':
        k2 = george.kernels.Matern32Kernel(length_scale**2)
    elif kernel == 'squaredexp':
        k2 = george.kernels.ExpSquaredKernel(length_scale**2)
    else:
        raise ValueError(f'"{kernel}" is not a valid kernel.')

    ker = k1*k2

    gp = george.GP(kernel=ker, fit_mean=True)
    # initial guess
    gp.compute(x, yerr)

    # optimization routine for hyperparameters
    p0 = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    results = scipy.optimize.minimize(neg_log_like, p0, jac=grad_neg_log_like,
                                            method="L-BFGS-B", options={'maxiter':30},
                                            bounds=bounds)
    gp.set_parameter_vector(results.x)

    step = 0.01/x_norm
    x_pred = np.arange(x_min, x_max+step, step)

    mean, var = gp.predict(y, x_pred, return_var=True)
    std = np.sqrt(var)

    return x_pred*x_norm, mean*y_norm, std*y_norm

def spline_mf_fit(x_data, y_data, yerr_data=0.0, x_edges=[1e3, 3e4], linear_extrap=True):
    """Fits a mangling function with a univariate spline.

    The package ``george`` is used for the gaussian process fit.

    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, default ``0.0``
        Dependent value errors.
    x_edges: array-like, default ``[1e3, 3e4]``
        Minimum and maximum x-axis values. These are used to extrapolate both edges.
    linear_extrap: bool, default ``True``
        Type of extrapolation for the edges. Linear if ``True``, free (gaussian process extrapolation) if ``False``.

    Returns
    -------
    x_pred : array
        Interpolated x-axis values.
    mean : array
        Interpolated  values.
    std : float
        Standard deviation (:math:`1\sigma`) of the interpolation. ``0.0`` for now - **STILL NEEDS TO BE IMPLEMENTED.**
    spline : obj
        Spline object used for the fit.

    """

    if linear_extrap:
        x, y, yerr = extrapolate_mangling_edges(np.copy(x_data), np.copy(y_data), np.copy(yerr_data), x_edges)
    else:
        x, y, yerr = np.copy(x_data), np.copy(y_data), np.copy(yerr_data)

    # compute spline
    s = len(y)
    spline = UnivariateSpline(x, y, k=3, s=s)

    step = 1
    x_min, x_max = x_edges
    x_pred = np.arange(x_min, x_max+step, step)
    mean, std = spline(x_pred), 0.0  # no uncertainty for now

    return x_pred, mean, std, spline


def gp_mf_fit(x_data, y_data, yerr_data=0.0, kernel='squaredexp', x_edges=[1e3, 3e4], linear_extrap=True):
    """Fits a mangling function with gaussian process.

    The package ``george`` is used for the gaussian process fit.

    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, default ``0.0``
        Dependent value errors.
    kernel : str, default ``squaredexp``
        Kernel to be used with the gaussian process. E.g., ``matern52``, ``matern32``, ``squaredexp``.
    x_edges: array-like, default ``[1e3, 3e4]``
        Minimum and maximum x-axis values. These are used to extrapolate both edges.
    linear_extrap: bool, default ``True``
        Type of extrapolation for the edges. Linear if ``True``, free (gaussian process extrapolation) if ``False``.

    Returns
    -------
    x_pred*x_norm : array
        Interpolated x-axis values.
    mean*y_norm : array
        Interpolated  values.
    std*y_norm : array
        Standard deviation (:math:`1\sigma`) of the interpolation.
    gp_results : dict
        Dictionary with the Gaussian Process object used for the fit and the normalisation terms.

    """

    if linear_extrap:
        x, y, yerr = extrapolate_mangling_edges(np.copy(x_data), np.copy(y_data), np.copy(yerr_data), x_edges)
    else:
        x, y, yerr = np.copy(x_data), np.copy(y_data), np.copy(yerr_data)

    # normalise the data for better results
    x_norm = 1e3
    x /= x_norm
    x_min, x_max = np.array(x_edges)/x_norm

    y_norm = y.max()
    y /= y_norm
    yerr /= y_norm

    mean_model = y.mean()

    var, length = np.var(y), 20  # fixed length to give smooth fits
    bounds_var, bounds_length = [(np.log(1e-8), np.log(1e4))], [(np.log(1e-6), np.log(1e4))]

    # a constant kernel is used to allow adding bounds
    k1 = george.kernels.ConstantKernel(np.log(var), bounds=bounds_var)

    if kernel == 'matern52':
        k2 = george.kernels.Matern52Kernel(length**2, metric_bounds=bounds_length)
    elif kernel == 'matern32':
        k2 = george.kernels.Matern32Kernel(length**2, metric_bounds=bounds_length)
    elif kernel == 'squaredexp':
        k2 = george.kernels.ExpSquaredKernel(length**2, metric_bounds=bounds_length)
    else:
        raise ValueError(f'"{kernel}" is not a valid kernel.')

    ker = k1*k2

    gp = george.GP(kernel=ker, mean=mean_model)
    gp.compute(x, yerr)

    step = 1/x_norm
    x_pred = np.arange(x_min, x_max+step, step)

    mean, var = gp.predict(y, x_pred, return_var=True)
    std = np.sqrt(var)

    gp_results = {'gp':partial(gp.predict, y), 'x_norm':x_norm, 'y_norm':y_norm}
    return x_pred*x_norm, mean*y_norm, std*y_norm, gp_results


def gp_2d_fit(x1_data, x2_data, y_data, yerr_data=0.0, kernel1='matern52', kernel2='matern52',
                var=None, length1=None, length2=None, x1_edges=None, x2_edges=None, optimization=True):
    """Fits light curves in 2D with gaussian process.

    The package ``george`` is used for the gaussian process fit.

    Parameters
    ----------
    x1_data : array
        First dimension of the x-axis grid.
    x2_data : array
        Second dimension of the x-axis grid.
    y_data : array
        Dependent values.
    yerr_data : array or float, default ``0.0``
        Dependent value errors.
    kernel1 : str, default ``matern52``
        Kernel to be used to fit the light curves with gaussian process. E.g., ``matern52``, ``matern32``, ``squaredexp``.
    kernel2 : str, default ``matern52``
        Kernel to be used in the wavelength axis when fitting in 2D with gaussian process. E.g., ``matern52``, ``matern32``, ``squaredexp``.
    var: float, default ``None``
        Variance of the kernel to be used.
    length1: float, default ``None``
        Length scale of the kernel to be used for ``x1_data``.
    length2: float, default ``None``
        Length scale of the kernel to be used for ``x2_data``.
    x1_edges: array-like, default ``None``
        Minimum and maximum ``x1_data`` values. These are used to extrapolate both edges.
    x2_edges: array-like, default ``None``
        Minimum and maximum ``x2_data`` values. These are used to extrapolate both edges.
    optimization: bool, default ``True``
        Whether or not to optimize the gaussian process hyperparameters. This is used to fit the light curves.

    Returns
    -------
    X_predict : array
        Interpolated 2D x-axis grid.
    mean : array
        Interpolated  values.
    std : array
        Standard deviation (:math:`1\sigma`) of the interpolation.
    gp_results : dict
        Dictionary with the Gaussian Process object used for the fit and the normalisation terms.
    """

    # define the objective function (negative log-likelihood in this case)
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    # and the gradient of the objective function
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    x1, x2  = np.copy(x1_data), np.copy(x2_data)
    y, yerr = np.copy(y_data), np.copy(yerr_data)

    # normalize data
    y_norm = y.max()
    x1_norm = 1e4
    x2_norm = 1e3

    y /= y_norm
    yerr /= y_norm
    x1 /= x1_norm
    x2 /= x2_norm

    X = np.array([x1, x2]).reshape(2, -1).T

    # define kernel
    kernels_dict = {'matern52':george.kernels.Matern52Kernel,
                    'matern32':george.kernels.Matern32Kernel,
                    'squaredexp':george.kernels.ExpSquaredKernel,
                    }
    assert kernel1 in kernels_dict.keys(), f'"{kernel1}" is not a valid kernel, choose one of the following ones: {list(kernels_dict.keys())}'
    assert kernel2 in kernels_dict.keys(), f'"{kernel2}" is not a valid kernel, choose one of the following ones: {list(kernels_dict.keys())}'

    # check if hyprerparameters were given
    if not var:
        var = np.var(y)
    if not length1:
        length1 = np.diff(x1).max()
    if not length2:
        length2 = np.diff(x2).max()

    ker1, ker2 = kernels_dict[kernel1], kernels_dict[kernel2]
    ker = var * ker1(length1**2, ndim=2, axes=0) * ker2(length2**2, ndim=2, axes=1)

    mean_function =  y.mean()
    gp = george.GP(kernel=ker, mean=mean_function, fit_mean=True)
    # initial guess
    gp.compute(X, yerr)

    # optimization routine for hyperparameters
    if optimization:
        p0 = gp.get_parameter_vector()
        results = scipy.optimize.minimize(neg_ln_like, p0, jac=grad_neg_ln_like, method="L-BFGS-B")
        gp.set_parameter_vector(results.x)

    # check edges
    if np.any(x1_edges):
        x1_edges = np.copy(x1_edges)
        x1_edges /= x1_norm
        x1_min, x1_max = x1_edges[0], x1_edges[-1]
    else:
        x1_min, x1_max = x1.min(), x1.max()
    if np.any(x2_edges):
        x2_edges = np.copy(x2_edges)
        x2_edges /= x2_norm
        x2_min, x2_max = x2_edges[0] - 200/x2_norm, x2_edges[-1] + 200/x2_norm  # extrapolate wavelength edges a bit further
    else:
        x2_min, x2_max = x2.min(), x2.max()

    x1_min = np.floor(x1_min*x1_norm)/x1_norm
    x1_max = np.ceil(x1_max*x1_norm)/x1_norm
    x2_min = np.floor(x2_min*x2_norm)/x2_norm
    x2_max = np.ceil(x2_max*x2_norm)/x2_norm
    step1 = 0.1/x1_norm  # in days/x1_norm
    step2 = 10/x2_norm  # in angstroms/x1_norm

    X_predict = np.array(np.meshgrid(np.arange(x1_min, x1_max+step1, step1),
                             np.arange(x2_min, x2_max+step2, step2))).reshape(2, -1).T

    mean, var = gp.predict(y, X_predict, return_var=True)
    std = np.sqrt(var)

    # de-normalize results
    X_predict *= np.array([x1_norm, x2_norm])
    mean *= y_norm
    std *= y_norm

    gp_results = {'gp':partial(gp.predict, y), 'x1_norm':x1_norm, 'x2_norm':x2_norm, 'y_norm':y_norm}

    return X_predict, mean, std, gp_results
