import unittest
import piscola
import numpy as np

class TestPiscola(unittest.TestCase):

    def test_lc_correction(self):

        sn = piscola.call_sn('03D1au')
        sn.fit_lcs()
        sn.mangle_sed(-1, 1)

if __name__ == '__main__':
    unittest.main()
