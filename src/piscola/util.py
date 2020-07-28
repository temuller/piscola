import numpy as np

def trim_filters(response):
    """
    Trim the leading and trailing zeros from a 1-D array or sequence, leaving
    one zero on each side. This is a modified version of numpy.trim_zeros.

    Parameters
    ----------
    response : 1-D array or sequence
        Input array.

    Returns
    -------
    The resulting indeces of trimming the input. The input data type is preserved.
    """

    first = 0
    for i in response:
        if i != 0.:
            if first == 0:
                first += 1  # to avoid filters with non-zero edges
            break
        else:
            first = first + 1

    last = len(response)
    for i in response[::-1]:
        if i != 0.:
            if last == len(response):
                last -= 1  # to avoid filters with non-zero edges
            break
        else:
            last = last - 1


    return first-1, last+1


def extrapolate_mangling_edges(x, y, yerr, x_edges, extra_extension=0.0):
    """"
    Extrapolates the edges of y acording to the giving edges in x (x_edges).
    """

    x_min, x_max = x_edges
    x_min -= extra_extension
    x_max += extra_extension

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
    try:
        yerr_min = np.sqrt(yerr[0]**2 + yerr[1]**2)
        yerr_max = np.sqrt(yerr[-1]**2 + yerr[-2]**2)
        yerr = np.r_[yerr_min, yerr, yerr_max]
    except:
        pass

    return x, y, yerr
