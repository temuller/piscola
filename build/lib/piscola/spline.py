from scipy.interpolate import UnivariateSpline
import numpy as np

def fit_spline(x_data, y_data, yerr_data=0.0, x_edges=None):
    """Fits data with a spline.
    
    NOTE: this function does not return the correct standard deviation
    
    Parameters
    ----------
    x_data : array
        Independent values.
    y_data : array
        Dependent values.
    yerr_data : array, int
        Dependent value errors.
    x_edges: array-like, default 'None'
        Minimum and maximum x-axis values. These are used to extrapolate both edges if 'mangling=1'.
        
    Returns
    -------
    Returns the interpolated independent and dependent values with the 1-sigma standard deviation.
    
    """
    
    x, y, yerr = np.copy(x_data), np.copy(y_data), np.copy(yerr_data)
    
    # anchor the edges
    x_min, x_max = x_edges
    pmin = np.polyfit(x[:2], y[:2], deg=1)
    pmax = np.polyfit(x[-2:], y[-2:], deg=1)
    
    y = np.r_[np.poly1d(pmin)(x_min), y, np.poly1d(pmax)(x_max)]
    x = np.r_[x_min, x, x_max]
    yerr_min = np.sqrt(yerr[0]**2 + yerr[1]**2)
    yerr_max = np.sqrt(yerr[-1]**2 + yerr[-2]**2)
    yerr = np.r_[yerr_min, yerr, yerr_max] 
    
    spl = UnivariateSpline(x, y, w=1/yerr, s=0)
    
    step = 10
    t = np.arange(x.min(), x.max()+step, step)
    
    mu = spl(t)
    std = mu*0.03
    
    return t, mu, std
