import unittest
from piscola.filters_class import MultiFilters
import matplotlib.pyplot as plt


class TestPiscola(unittest.TestCase):
    def test_plot(self):
        filters = MultiFilters(["ps1_g"], ["AB"])
        plt.ion()
        filters.plot_filters()

    def test_remove_filter(self):
        filters = MultiFilters(["ps1_g"], ["AB"])
        filters.remove_filter("ps1_g")

    def test_pivot_wavelength(self):
        filters = MultiFilters(["ps1_g"], ["AB"])
        filters.calc_pivot()


if __name__ == "__main__":
    unittest.main()
