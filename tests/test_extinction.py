import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
import piscola
from piscola.extinction_correction import (
    calculate_ebv,
    redden,
    deredden,
    extinction_filter,
    extinction_curve,
)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestPiscola(unittest.TestCase):
    def test_calculate_ebv(self):
        ra, dec = 30, 30

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            ebv_SF2011 = calculate_ebv(ra, dec, scaling=0.86)
            ebv_SFD1998 = calculate_ebv(ra, dec, scaling=1)

        # reference values from https://irsa.ipac.caltech.edu/cgi-bin/bgTools/nph-bgExec
        ref_ebv_SF2011 = 0.0431  # mean value S & F (2011)
        ref_ebv_SFD1998 = 0.0502  # mean value SFD (1998)

        err_msg = (
            "Difference in E(B-V) between calculated and reference value is too large."
        )
        np.testing.assert_allclose(
            ebv_SF2011, ref_ebv_SF2011, rtol=0.02, err_msg=err_msg
        )
        np.testing.assert_allclose(
            ebv_SFD1998, ref_ebv_SFD1998, rtol=0.02, err_msg=err_msg
        )

    def test_reddening_laws(self):
        ra, dec = 30, 30

        wave = np.arange(1000, 20000)
        flux = np.ones_like(wave)

        for reddening_law in [
            "fitzpatrick99",
            "ccm89",
            "odonnell94",
            "calzetti00",
            "fm07",
        ]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                reddened_flux = redden(wave, flux, ra, dec, reddening_law=reddening_law)
            self.assertTrue(reddened_flux.mean() < flux.mean())

    def test_ebv(self):
        ra, dec = 30, 30

        wave = np.arange(1000, 20000)
        flux = np.ones_like(wave)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            reddened_flux_maps = redden(wave, flux, ra, dec)

            ebv = calculate_ebv(ra, dec)
            reddened_flux_ebv = redden(wave, flux, ra, dec, ebv=ebv)

        self.assertTrue(reddened_flux_maps.mean() == reddened_flux_ebv.mean())

    def test_deredden(self):
        ra, dec = 30, 30

        wave = np.arange(1000, 20000)
        flux = np.ones_like(wave)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            reddened_flux = redden(wave, flux, ra, dec)
            dereddened_flux = deredden(wave, reddened_flux, ra, dec)

        self.assertTrue(dereddened_flux.mean() == flux.mean())

    def test_visual_extinction(self):
        pisco_path = piscola.__path__[0]
        ra, dec = 30, 30

        filter_file = os.path.join(pisco_path, "filters", "Bessell", "Bessell_V.dat")
        filter_wave, filter_response = np.loadtxt(filter_file).T
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            A_V = extinction_filter(filter_wave, filter_response, ra, dec, r_v=3.1)
        ref_A_V = (
            0.1368  # from https://irsa.ipac.caltech.edu/cgi-bin/bgTools/nph-bgExec
        )

        err_msg = (
            "Difference in A_V between calculated and reference value is too large."
        )
        np.testing.assert_allclose(A_V, ref_A_V, rtol=0.03, err_msg=err_msg)

    def test_extinction_curve(self):
        ra, dec = 30, 30
        plt.ion()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            extinction_curve(ra, dec)


if __name__ == "__main__":
    unittest.main()
