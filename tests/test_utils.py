import unittest
from piscola import __path__
from piscola.filter_utils import calc_zp
import numpy as np
import os

class TestPiscola(unittest.TestCase):

    def test_utils(self):

        path = __path__[0]
        filter_wave, filter_trans = np.loadtxt(os.path.join(path, 'filters/Bessell/Bessell_B.dat')).T
        # VEGA needs to be implemented if necessary in the future
        for mag_sys in ['BD17']:
            zp = calc_zp(filter_wave, filter_trans, 'photon', mag_sys, 'Bessell_B')

        filter_wave, filter_trans = np.loadtxt(os.path.join(path, 'filters/SDSS/sdss_g.dat')).T
        zp = calc_zp(filter_wave, filter_trans, 'photon', 'AB', 'sdss_g')

if __name__ == '__main__':
    unittest.main()
