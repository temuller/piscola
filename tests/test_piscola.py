import unittest
import piscola as pisco
import numpy as np

class TestPiscola(unittest.TestCase):
        
    def test_lc_correction(self):
        
        sn = pisco.sn_file('sn2001cz', directory='../../data/')
        sn.normalize_data()
        sn.fit_lcs()
        sn.mangle_sed(-5, 15)
        sn.calculate_lc_params()

        tmax = sn.tmax
        mb = sn.lc_parameters['mb']
        dm15 = sn.lc_parameters['dm15']
        color = sn.lc_parameters['color']

        self.assertEqual(np.round(tmax, 0), 52103, "Incorrect tmax estimation")
        self.assertEqual(np.round(mb, 1), 15.1, "Incorrect mb estimation")
        self.assertEqual(np.round(dm15, 1), 1.0, "Incorrect dm15 estimation")
        self.assertEqual(np.round(color, 1), 0.1, "Incorrect color estimation")

if __name__ == '__main__':
    unittest.main()
