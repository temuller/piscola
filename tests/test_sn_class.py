import unittest
import piscola
import matplotlib.pyplot as plt

import warnings


class TestPiscola(unittest.TestCase):
    def test_plot(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            sn = piscola.call_sn("data/03D1au.dat")
            plt.ion()
            sn.plot_lcs()

    def test_fit_and_outputs(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            sn = piscola.call_sn("data/03D1au.dat")
            sn.fit()

            plt.ion()
            sn.plot_fits()

            sn.export_fits()
            sn.export_restframe_lcs()


if __name__ == "__main__":
    unittest.main()
