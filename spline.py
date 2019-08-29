from scipy.interpolate import UnivariateSpline
import numpy as np

def fit_spline(x_data, y_data, yerr_data=0.0, x_edges=None):
    """Fits data with a spline.
    
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

    step = 10
    t = np.arange(x.min(), x.max()+step, step)
    Y = [np.random.normal(y, yerr) for i in range(1000)]
    splines_list = list()
    
    for y_spline in Y:
        spl = UnivariateSpline(x, y_spline, s=0, k=2)
        splines_list.append(spl(t))
        
    splines_array = np.asarray(splines_list)
    mu = np.average(splines_array, axis=0)
    std = np.std(splines_array, axis=0)
    
    return t, mu, std
