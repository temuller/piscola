import numpy as np
import george
import scipy
import emcee

from .util import extrapolate_mangling_edges

class gaussian_lcMeanModel(george.modeling.Model):
    """Gaussian light curve model."""
    parameter_names = ("A", "mu", "log_sigma2")

    def get_value(self, t):
        return self.A * np.exp(-0.5*(t-self.mu)**2 * np.exp(-self.log_sigma2))

    # This method is to compute the gradient of the objective.
    def compute_gradient(self, t):
        e = 0.5*(t-self.mu)**2 * np.exp(-self.log_sigma2)
        dA = np.exp(-e)
        dmu = self.A * dA * (t-self.mu) * np.exp(-self.log_sigma2)
        dlog_s2 = self.A * dA * e
        return np.array([dA, dmu, dlog_s2])

class bazin_lcMeanModel(george.modeling.Model):
    """Bazin et al. (2011) light curve model."""
    parameter_names = ("A", "t0", "tf", "tr")

    def get_value(self, t):
        np.seterr(all='ignore')
        bazin_model = self.A * np.exp(-(t-self.t0)/self.tf) / (1 + np.exp(-(t-self.t0)/self.tr))
        bazin_model = np.nan_to_num(bazin_model, nan=1e-6)  # prevents to output an error
        np.seterr(all='warn')

        return bazin_model

    # This method is to compute the gradient of the objective.
    def compute_gradient(self, t):
        np.seterr(all='ignore')
        Tf = (t-self.t0)/self.tf
        Tr = (t-self.t0)/self.tr
        B = np.exp(-Tf)
        C = 1 + np.exp(-Tr)

        dA = B/C
        dtf = A*B/C * Tf/self.tf
        dtr = A*B/C**2 * Tf/self.tr * np.exp(-Tr)
        dt0 = (np.exp(-t/self.tf + self.t0/self.tf + t/self.tr) *
                    (self.tr*(np.exp(t/self.tr) + np.exp(self.t0/self.tr)) - self.tr*np.exp(self.t0/self.tr)) /
                    (self.tr*self.tf*(np.exp(t/self.tr) + np.exp(self.t0/self.tr))**2)
                )
        np.seterr(all='warn')

        return np.array([dA, dt0, dtf, dtr])

class zheng_lcMeanModel(george.modeling.Model):
    """Zheng et al. (2018) light curve model."""
    parameter_names = ("A", "t0", "tb", "ar", "ad", "s")

    def get_value(self, t):
        np.seterr(all='ignore')
        Tb = (t-self.t0)/self.tb
        zheng_model = self.A * Tb**self.ar * (1 + Tb**(self.s*self.ad))**(-2/self.s)
        zheng_model = np.nan_to_num(zheng_model, nan=1e-6)  # prevents to output an error
        np.seterr(all='warn')

        return zheng_model

    # This method is to compute the gradient of the objective.
    def compute_gradient(self, t):
        np.seterr(all='ignore')
        Tb = (t-self.t0)/self.tb
        Tb_ar = Tb**self.ar
        Tb_sad = Tb**(self.s*self.ad)

        dA = Tb_ar * (1 + Tb_sad)**(-2/self.s)
        dt0 = ((2*self.A*self.ad * (Tb_sad + 1)**(-2/self.s - 1) * Tb_sad*Tb_ar/Tb)/self.tb -
                            (self.A*self.ar*Tb_ar/Tb * (Tb_sad + 1)**(-2/self.s))/self.tb
               )
        dtb = dt0*Tb
        dar = self.A * Tb_ar * np.log(Tb) * (Tb_sad + 1)**(-2/self.s)
        dad = -2*self.A * np.log(Tb) * (Tb_sad + 1)**(-2/self.s - 1) * Tb_sad*Tb_ar
        ds = self.A * Tb_ar * (Tb_sad + 1)**(-2/self.s) * (2*np.log(Tb_sad + 1)/self.s**2
                                                       - 2*self.ad*np.log(Tb)*Tb_sad/(self.s*(Tb_sad + 1)))
        np.seterr(all='warn')

        return np.array([dA, dt0, dtb, dar, dad, ds])

class mangling_MeanModel(george.modeling.Model):
    """Polynomial mangling function model with 3 degrees and no constant parameter."""
    parameter_names = ("c1", "c2", "c3")

    def get_value(self, t):
        return self.c1*t + self.c2*t**2 + self.c3*t**3

    def compute_gradient(self, t):
        dc1 = t
        dc2 = t**2
        dc3 = t**3
        return np.array([dc1, dc2, dc3])


def gp_lc_fit(x_data, y_data, yerr_data=0.0, kernel='matern52', gp_mean='mean'):
    """Fits a light curve with gaussian process.

    The package 'george' is used for the gaussian process fit.

    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, int
        Dependent value errors.
    kernel : str, default 'squaredexp'
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
        If left default, 'matern52' is used to fit light curves and 'squaredexp' for the mangling function.
    mangling: bool, default 'False'
        If 'True', the fit is set to adjust the mangling function.
    x_edges: array-like, default 'None'
        Minimum and maximum x-axis values. These are used to extrapolate both edges if 'mangling==True'.

    Returns
    -------
    Returns the interpolated independent and dependent values with the 1-sigma standard deviation.

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

    # choose GP mean model with the respective bounds
    if gp_mean=='mean':
        mean_model = y.mean()  # constant model
    elif gp_mean=='gaussian':
        A, mu, log_sigma2 = y.max(), x[y==y.max()][0], np.log(10)
        mean_bounds = {'A':(0.1, 100),
                       'mu':(mu-50, mu+50),
                       'log_sigma2':(np.log(10), np.log(60)),
                       }
        mean_model = gaussian_lcMeanModel(A=A, mu=mu, log_sigma2=log_sigma2, bounds=mean_bounds)
    elif gp_mean=='bazin':
        A, tf, tr = y.max(), 40, 20
        t0 = 20 + tr*np.log(tf/tr-1)
        mean_bounds = {'A':(0.1, 100),
                       't0':(t0-50, t0+50),
                       'tf':(tf-35, tf+40),
                       'tr':(tr-15, tr+20),
                       }
        mean_model = bazin_lcMeanModel(A=A, t0=t0, tf=tf, tr=tr, bounds=mean_bounds)
    elif gp_mean=='zheng':
        A, t0, tb, ar, ad, s = y.max(), x[y==y.max()][0]-20, 20, 2, 2.5, 1.5
        mean_bounds = {'A':(0.1, 100),
               't0':(t0-50, t0+50),
               'tb':(tb-15, tb+20),
               'ar':(ar-1.8, ar+3.0),
               'ad':(ad-2.3, ad+3.5),
               's':(s-1.3, s+3.0),
               }
        mean_model = zheng_lcMeanModel(A=A, t0=t0, tb=tb, ar=ar, ad=ad, s=s, bounds=mean_bounds)
    else:
        raise ValueError(f'"{gp_mean}" is not a valid gaussian-process mean function for light-curve fitting.')

    var, length_scale = np.var(y), np.diff(x).max()
    bounds_var, bounds_length = [(np.log(1e-6), np.log(10))], [(np.log(1e-8), np.log(1e2))]

    # a constant kernel is used to allow adding bounds
    k1 = george.kernels.ConstantKernel(np.log(var), bounds=bounds_var)

    if kernel == 'matern52':
        k2 = george.kernels.Matern52Kernel(length_scale**2, metric_bounds=bounds_length)
    elif kernel == 'matern32':
        k2 = george.kernels.Matern32Kernel(length_scale**2, metric_bounds=bounds_length)
    elif kernel == 'squaredexp':
        k2 = george.kernels.ExpSquaredKernel(length_scale**2, metric_bounds=bounds_length)
    else:
        raise ValueError(f'"{kernel}" is not a valid kernel.')

    ker = k1*k2

    gp = george.GP(kernel=ker, mean=mean_model, fit_mean=True)
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

    mu, var = gp.predict(y, x_pred, return_var=True)
    std = np.sqrt(var)

    return x_pred*x_norm, mu*y_norm, std*y_norm


def gp_mf_fit(x_data, y_data, yerr_data=0.0, kernel='squaredexp', gp_mean='poly', x_edges=[1e3, 3e4]):
    """Fits a mangling function with gaussian process.

    The package 'george' is used for the gaussian process fit.

    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, int
        Dependent value errors.
    kernel : str, default 'squaredexp'
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
        If left default, 'matern52' is used to fit light curves and 'squaredexp' for the mangling function.
    mangling: bool, default 'False'
        If 'True', the fit is set to adjust the mangling function.
    x_edges: array-like, default 'None'
        Minimum and maximum x-axis values. These are used to extrapolate both edges if 'mangling==True'.

    Returns
    -------
    Returns the interpolated independent and dependent values with the 1-sigma standard deviation.

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

    x, y, yerr = extrapolate_mangling_edges(np.copy(x_data), np.copy(y_data), np.copy(yerr_data), x_edges)

    # normalise the data for better results
    x_norm = 1e3
    x /= x_norm
    x_min, x_max = x.min(), x.max()

    y_norm = y.max()
    y /= y_norm
    yerr /= y_norm

    # choose GP mean model with the respective bounds
    if gp_mean=='poly':
        poly = np.poly1d(np.polyfit(x, y, 3))
        c1, c2, c3 = poly.coeffs[2], poly.coeffs[1], poly.coeffs[0]
        mean_bounds = {'c1':(-1e2, 1e2), 'c2':(-1e2, 1e2), 'c3':(-1e2, 1e2)}
        mean_model = mangling_MeanModel(c1=c1, c2=c2, c3=c3, bounds=mean_bounds)
    elif gp_mean=='mean':
        mean_model = y.mean()  # constant model
    else:
        raise ValueError(f'"{gp_mean}" is not a valid gaussian-process mean function for mangling-function fitting.')

    var, length = np.var(y), np.diff(x).max()
    bounds_var, bounds_length = [(np.log(1e-8), np.log(1e4))], [(np.log(1e-6), np.log(1e4))]

    # a constant kernel is used to allow adding bounds
    k1 = george.kernels.ConstantKernel(np.log(var),bounds=bounds_var)

    if kernel == 'matern52':
        k2 = george.kernels.Matern52Kernel(length**2, metric_bounds=bounds_length)
    elif kernel == 'matern32':
        k2 = george.kernels.Matern32Kernel(length**2, metric_bounds=bounds_length)
    elif kernel == 'squaredexp':
        k2 = george.kernels.ExpSquaredKernel(length**2, metric_bounds=bounds_length)
    else:
        raise ValueError(f'"{kernel}" is not a valid kernel.')

    ker = k1*k2

    gp = george.GP(kernel=ker, mean=mean_model, fit_mean=True)
    # initial guess
    gp.compute(x, yerr)

    # optimization routine for hyperparameters
    # p0 = gp.get_parameter_vector()
    # bounds = gp.get_parameter_bounds()
    # results = scipy.optimize.minimize(neg_log_like, p0, jac=grad_neg_log_like,
    #                                         method="L-BFGS-B", options={'maxiter':30},
    #                                         bounds=bounds)
    # gp.set_parameter_vector(results.x)

    step = 1/x_norm
    x_pred = np.arange(x_min, x_max+step, step)

    mu, var = gp.predict(y, x_pred, return_var=True)
    std = np.sqrt(var)

    return x_pred*x_norm, mu*y_norm, std*y_norm


def fit_gp(x_data, y_data, yerr_data=1e-8, kernel=None, gp_mean='mean', x_edges=None, free_extrapolation=False):
    """Fits data with gaussian process.

    The package 'george' is used for the gaussian process fit.

    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, int
        Dependent value errors.
    kernel : str, default 'squaredexp'
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
        If left default, 'matern52' is used to fit light curves and 'squaredexp' for the mangling function.
    mangling: bool, default 'False'
        If 'True', the fit is set to adjust the mangling function.
    x_edges: array-like, default 'None'
        Minimum and maximum x-axis values. These are used to extrapolate both edges if 'mangling==True'.

    Returns
    -------
    Returns the interpolated independent and dependent values with the 1-sigma standard deviation.

    """

    if gp_mean=='gaussian':
        class lcMeanModel(george.modeling.Model):
            """Gaussian light curve model."""
            parameter_names = ("A", "mu", "log_sigma2")

            def get_value(self, t):
                return self.A * np.exp(-0.5*(t-self.mu)**2 * np.exp(-self.log_sigma2))

            # This method is to compute the gradient of the objective function below.
            def compute_gradient(self, t):
                e = 0.5*(t-self.mu)**2 * np.exp(-self.log_sigma2)
                dA = np.exp(-e)
                dmu = self.A * dA * (t-self.mu) * np.exp(-self.log_sigma2)
                dlog_s2 = self.A * dA * e
                return np.array([dA, dmu, dlog_s2])

    elif gp_mean=='bazin':
        class lcMeanModel(george.modeling.Model):
            """Bazin et al. (2011) light curve model."""
            parameter_names = ("A", "t0", "tf", "tr")

            def get_value(self, t):
                np.seterr(all='ignore')
                bazin_model = self.A * np.exp(-(t-self.t0)/self.tf) / (1 + np.exp(-(t-self.t0)/self.tr))
                bazin_model = np.nan_to_num(bazin_model, nan=1e-6)  # prevents to output an error
                np.seterr(all='warn')

                return bazin_model

            def compute_gradient(self, t):
                np.seterr(all='ignore')
                Tf = (t-self.t0)/self.tf
                Tr = (t-self.t0)/self.tr
                B = np.exp(-Tf)
                C = 1 + np.exp(-Tr)

                dA = B/C
                dtf = A*B/C * Tf/self.tf
                dtr = A*B/C**2 * Tf/self.tr * np.exp(-Tr)
                dt0 = (np.exp(-t/self.tf + self.t0/self.tf + t/self.tr) *
                            (self.tr*(np.exp(t/self.tr) + np.exp(self.t0/self.tr)) - self.tr*np.exp(self.t0/self.tr)) /
                            (self.tr*self.tf*(np.exp(t/self.tr) + np.exp(self.t0/self.tr))**2)
                        )
                np.seterr(all='warn')

                return np.array([dA, dt0, dtf, dtr])

    elif gp_mean=='zheng':
        class lcMeanModel(george.modeling.Model):
            """Zheng et al. (2018) light curve model."""
            parameter_names = ("A", "t0", "tb", "ar", "ad", "s")

            def get_value(self, t):
                np.seterr(all='ignore')
                Tb = (t-self.t0)/self.tb
                zheng_model = self.A * Tb**self.ar * (1 + Tb**(self.s*self.ad))**(-2/self.s)
                zheng_model = np.nan_to_num(zheng_model, nan=1e-6)  # prevents to output an error
                np.seterr(all='warn')

                return zheng_model

            def compute_gradient(self, t):
                np.seterr(all='ignore')
                Tb = (t-self.t0)/self.tb
                Tb_ar = Tb**self.ar
                Tb_sad = Tb**(self.s*self.ad)

                dA = Tb_ar * (1 + Tb_sad)**(-2/self.s)
                dt0 = ((2*self.A*self.ad * (Tb_sad + 1)**(-2/self.s - 1) * Tb_sad*Tb_ar/Tb)/self.tb -
                                    (self.A*self.ar*Tb_ar/Tb * (Tb_sad + 1)**(-2/self.s))/self.tb
                       )
                dtb = dt0*Tb
                dar = self.A * Tb_ar * np.log(Tb) * (Tb_sad + 1)**(-2/self.s)
                dad = -2*self.A * np.log(Tb) * (Tb_sad + 1)**(-2/self.s - 1) * Tb_sad*Tb_ar
                ds = self.A * Tb_ar * (Tb_sad + 1)**(-2/self.s) * (2*np.log(Tb_sad + 1)/self.s**2
                                                               - 2*self.ad*np.log(Tb)*Tb_sad/(self.s*(Tb_sad + 1)))
                np.seterr(all='warn')

                return np.array([dA, dt0, dtb, dar, dad, ds])

    class manglingMeanModel(george.modeling.Model):
        """Polynomial mangling function model with 3 degrees and no constant parameter."""
        parameter_names = ("c1", "c2", "c3")

        def get_value(self, t):
            return self.c1*t + self.c2*t**2 + self.c3*t**3

        def compute_gradient(self, t):
            dc1 = t
            dc2 = t**2
            dc3 = t**3
            return np.array([dc1, dc2, dc3])

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

    if kernel is None:
        kernel = 'matern52'

    x, y, yerr = np.copy(x_data), np.copy(y_data), np.copy(yerr_data)

    if np.any(x_edges) and not free_extrapolation:
        x_min, x_max = x_edges
        x_min -= 200
        x_max += 200

        # linear extrapolation
        pmin = np.polyfit(x[:2], y[:2], deg=1)
        pmax = np.polyfit(x[-2:], y[-2:], deg=1)
        y_left, y_right = np.poly1d(pmin)(x_min), np.poly1d(pmax)(x_max)

        # avoid negative values
        if y_left<0:
            y_left=0
        if y_right<0:
            y_right=0
        y = np.r_[y_left, y, y_right]
        x = np.r_[x_min, x, x_max]
        yerr_min = np.sqrt(yerr[0]**2 + yerr[1]**2)
        yerr_max = np.sqrt(yerr[-1]**2 + yerr[-2]**2)
        yerr = np.r_[yerr_min, yerr, yerr_max]

        x_norm = 1e3
        x /= x_norm
        x_min, x_max = x.min(), x.max()

    elif np.any(x_edges) and free_extrapolation:
        # let the GP extrapolate
        x_norm = 1e3
        x /= x_norm
        x_min /= x_norm
        x_max /= x_norm

    else:
        # not for mangling, only light curves with time in phase
        x_norm = 1
        x_min, x_max = x.min(), x.max()

    # normalise the data for better results
    y_norm = y.max()
    y /= y_norm
    yerr /= y_norm

    if x_edges is None:
        # choose GP mean model with the respective bounds
        if gp_mean=='mean':
            mean_model = y.mean()

        elif gp_mean=='gaussian':
            A, mu, log_sigma2 = y.max(), x[y==y.max()][0], np.log(10)
            mean_bounds = {'A':(0.1, 100),
                           'mu':(mu-50, mu+50),
                           'log_sigma2':(np.log(10), np.log(60)),
                           }
            mean_model = lcMeanModel(A=A, mu=mu, log_sigma2=log_sigma2, bounds=mean_bounds)

        elif gp_mean=='bazin':
            A, tf, tr = y.max(), 40, 20
            t0 = 20 + tr*np.log(tf/tr-1)
            mean_bounds = {'A':(0.1, 100),
                           't0':(t0-50, t0+50),
                           'tf':(tf-35, tf+40),
                           'tr':(tr-15, tr+20),
                           }
            mean_model = lcMeanModel(A=A, t0=t0, tf=tf, tr=tr, bounds=mean_bounds)

        elif gp_mean=='zheng':
            A, t0, tb, ar, ad, s = y.max(), x[y==y.max()][0]-20, 20, 2, 2.5, 1.5
            mean_bounds = {'A':(0.1, 100),
                   't0':(t0-50, t0+50),
                   'tb':(tb-15, tb+20),
                   'ar':(ar-1.8, ar+3.0),
                   'ad':(ad-2.3, ad+3.5),
                   's':(s-1.3, s+3.0),
                   }
            mean_model = lcMeanModel(A=A, t0=t0, tb=tb, ar=ar, ad=ad, s=s, bounds=mean_bounds)

    else:
        poly = np.poly1d(np.polyfit(x, y, 3))
        c1, c2, c3 = poly.coeffs[2], poly.coeffs[1], poly.coeffs[0]
        mean_bounds = {'c1':(-1e2, 1e2), 'c2':(-1e2, 1e2), 'c3':(-1e2, 1e2)}
        mean_model = manglingMeanModel(c1=c1, c2=c2, c3=c3, bounds=mean_bounds)

    var, length_scale = np.var(y), np.diff(x).max()
    bounds_var, bounds_length = [(np.log(1e-6), np.log(10))], [(np.log(1), np.log(1e4))]

    # a constant kernel is used to allow adding bounds
    k1 = george.kernels.ConstantKernel(np.log(var), bounds=bounds_var)

    if kernel == 'matern52':
        k2 = george.kernels.Matern52Kernel(length_scale**2, metric_bounds=bounds_length)
    elif kernel == 'matern32':
        k2 = george.kernels.Matern32Kernel(length_scale**2, metric_bounds=bounds_length)
    elif kernel == 'squaredexp':
        k2 = george.kernels.ExpSquaredKernel(length_scale**2, metric_bounds=bounds_length)
    else:
        raise ValueError(f'"{kernel}" is not a valid kernel.')

    ker = k1*k2
    #ker.freeze_parameter("k1:log_constant")
    #ker.freeze_parameter("k2:metric:log_M_0_0")
    # ['k1:log_constant', 'k2:metric:log_M_0_0']

    gp = george.GP(kernel=ker, mean=mean_model, fit_mean=True)
    # initial guess
    gp.compute(x, yerr)

    # optimization routine for hyperparameters
    p0 = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    results = scipy.optimize.minimize(neg_log_like, p0, jac=grad_neg_log_like,
                                            method="L-BFGS-B", options={'maxiter':30},
                                            bounds=bounds)
    gp.set_parameter_vector(results.x)

    step = 1e-3
    x_pred = np.arange(x_min, x_max+step, step)

    mu, var = gp.predict(y, x_pred, return_var=True)
    std = np.sqrt(var)

    return x_pred*x_norm, mu*y_norm, std*y_norm


def gp_2d_fit(x1_data, x2_data, y_data, yerr_data=0.0, kernel1='matern52', kernel2='squaredexp',
                var=None, length1=None, length2=None, x1_edges=None, x2_edges=None, optimization=1, use_mcmc=0):
    """Fits data with gaussian process.
    The package 'george' is used for the gaussian process fit.
    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, int
        Dependent value errors.
    kernel : str, default 'squaredexp'
        Kernel to be used with the gaussian process. Possible choices are: 'matern52', 'matern32', 'squaredexp'.
        If left default, 'matern52' is used to fit light curves and 'squaredexp' for the mangling function.
    mangling: bool, default 'False'
        If 'True', the fit is set to adjust the mangling function.
    x_edges: array-like, default 'None'
        Minimum and maximum x-axis values. These are used to extrapolate both edges if 'mangling==True'.
    Returns
    -------
    Returns the interpolated independent and dependent values with the 1-sigma standard deviation.
    """

    # define the objective function (negative log-likelihood in this case)
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    # and the gradient of the objective function
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    # for mcmc
    def lnprob(p):
        gp.set_parameter_vector(p)
        return gp.log_likelihood(y, quiet=True) + gp.log_prior()

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
    #ker.freeze_parameter('k1:k2:metric:log_M_0_0')
    #ker.freeze_parameter('k2:metric:log_M_0_0')

    mean_function =  y.mean()
    gp = george.GP(kernel=ker, mean=mean_function, fit_mean=True)
    # initial guess
    gp.compute(X, yerr)

    # optimization routine for hyperparameters
    if optimization:
        if use_mcmc:
            initial = gp.get_parameter_vector()
            ndim, nwalkers = len(initial), 32
            p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
            # Running burn-in...
            p0, _, _ = sampler.run_mcmc(p0, 500)
            sampler.reset()
            # Running production...
            sampler.run_mcmc(p0, 1000)
            samples = sampler.flatchain
            p_final = np.mean(samples, axis=0)
            gp.set_parameter_vector(p_final)
        else:
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
        x2_min, x2_max = x2_edges[0], x2_edges[-1]
    else:
        x2_min, x2_max = x2.min(), x2.max()

    x1_min = np.floor(x1_min*x1_norm)/x1_norm
    x1_max = np.ceil(x1_max*x1_norm)/x1_norm
    x2_min = np.floor(x2_min*x2_norm)/x2_norm
    x2_max = np.ceil(x2_max*x2_norm)/x2_norm
    step1 = 0.1/x1_norm
    step2 = 10/x2_norm

    X_predict = np.array(np.meshgrid(np.arange(x1_min, x1_max+step1, step1),
                             np.arange(x2_min, x2_max+step2, step2))).reshape(2, -1).T

    mu, var = gp.predict(y, X_predict, return_var=True)
    std = np.sqrt(var)

    # de-normalize results
    X_predict *= np.array([x1_norm, x2_norm])
    mu *= y_norm
    std *= y_norm

    return X_predict, mu, std
