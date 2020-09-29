import unittest
from piscola.gaussian_process import gp_lc_fit
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

class TestPiscola(unittest.TestCase):
    
    def test_gp_lc_fit(self):
        
        # create mock data
        x = np.arange(-10, 10, 0.5) 

        mu, sig, sig_noise = 0, 5, 0.5
        noise = np.random.normal(0, sig_noise, len(x))
        y = 100*norm.pdf(x, mu, sig) + noise
        yerr = sig_noise
        
        for gp_mean in ['mean', 'gaussian', 'bazin', 'zheng']:
            x_gp, y_gp, yerr_gp = gp_lc_fit(x, y, yerr, 'matern52', gp_mean)
                    
if __name__ == '__main__':
    unittest.main()

