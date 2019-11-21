import numpy as np
import george
import scipy

def fit_gp(x_data, y_data, yerr_data=0.0, kernel=None, mangling=False, x_edges=None):
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
     
    if kernel is None and mangling:
        kernel = 'squaredexp'
    elif kernel is None:
        kernel = 'matern52'

    x, y, yerr = np.copy(x_data), np.copy(y_data), np.copy(yerr_data)    
    x_norm = 1
    
    if mangling:
        x_min, x_max = x_edges
        x_min -= 200
        x_max += 200
        pmin = np.polyfit(x[:2], y[:2], deg=1)
        pmax = np.polyfit(x[-2:], y[-2:], deg=1)
        
        y_left, y_right = np.poly1d(pmin)(x_min), np.poly1d(pmax)(x_max)
        # avoid negative values
        if y_left<0:
            y_left=0
        if y_right<0:
            y_right=0
        y = np.r_[y_left, y, y_right]  # gradient extrapolation
        #y = np.r_[y[0], y, y[-1]]  # flat extrapolation
        x = np.r_[x_min, x, x_max]
        yerr_min = np.sqrt(yerr[0]**2 + yerr[1]**2)
        yerr_max = np.sqrt(yerr[-1]**2 + yerr[-2]**2)
        yerr = np.r_[yerr_min, yerr, yerr_max]
        
        x_norm = 1e3
                
    # normalise the data for better results
    y_norm = y.max()
    y /= y_norm 
    yerr /= y_norm
    mean_function = y.min()
    x /= x_norm
    
    ###### mean function for gp #######
    if mangling and len(x)>2:
        mean_function = y.mean()
    ###################################
    
    var, length = np.var(y), np.diff(x).max()
        
    if kernel == 'matern52':
        k = var * george.kernels.Matern52Kernel(length)
        
    elif kernel == 'matern32':
        k = var * george.kernels.Matern32Kernel(length)
        
    elif kernel == 'squaredexp':
        k = var * george.kernels.ExpSquaredKernel(length)
        
    else:
        raise ValueError(f'"{kernel}" is not a valid kernel.')

    gp = george.GP(kernel=k, solver=george.HODLRSolver, mean=mean_function)
    # initial guess
    gp.compute(x, yerr)
    
    # optimization routine for hyperparameters
    p0 = gp.get_parameter_vector()
    results = scipy.optimize.minimize(neg_ln_like, p0, jac=grad_neg_ln_like)
    gp.set_parameter_vector(results.x)
    
    step = 1e-3
    t = np.arange(x.min(), x.max()+step, step)
    
    mu, var = gp.predict(y, t, return_var=True)
    std = np.sqrt(var)
    
    return t*x_norm, mu*y_norm, std*y_norm
