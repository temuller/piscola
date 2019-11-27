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
            break
        else:
            first = first + 1

    last = len(response)
    for i in response[::-1]:
        if i != 0.:
            break
        else:
            last = last - 1

    return first-1, last+1
