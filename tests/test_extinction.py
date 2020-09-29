import unittest
from piscola.extinction_correction import calculate_ebv, redden, deredden
import numpy as np

class TestPiscola(unittest.TestCase):
    
    def test_mw_ebv(self):
        
        ra, dec = 0.0, 180.0
        ebv = calculate_ebv(ra, dec)
        
        self.assertTrue(ebv > 0.0)
        
    def test_reddening(self):
        
        ra, dec = 0.0, 180.0
        wave, flux = np.array([4000.0]), np.array([1000.0])
        
        for reddening_law in ['fitzpatrick99', 'ccm89']:
            redden_flux = redden(wave, flux, ra, dec, reddening_law=reddening_law)
            self.assertTrue(redden_flux[0] < flux[0])
            
    def test_dereddening(self):
        
        ra, dec = 0.0, 180.0
        wave, flux = np.array([4000.0]), np.array([1000.0])
        
        for reddening_law in ['fitzpatrick99', 'ccm89']:
            deredden_flux = deredden(wave, flux, ra, dec, reddening_law=reddening_law)
            self.assertTrue(deredden_flux[0] > flux[0])
            
if __name__ == '__main__':
    unittest.main()