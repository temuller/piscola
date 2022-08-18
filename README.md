# PISCOLA: Python for Intelligent Supernova-COsmology Light-curve Analysis

**Supernova light-curve fitting code in python**

Although the main purpose of PISCOLA is to fit type Ia supernovae, it can be used to fit other types of supernovae or even other transients.

[![repo](https://img.shields.io/badge/GitHub-temuller%2Fpiscola-blue.svg?style=flat)](https://github.com/temuller/piscola)
[![documentation status](https://readthedocs.org/projects/piscola/badge/?version=latest&style=flat)](https://piscola.readthedocs.io/en/latest/?badge=latest)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/temuller/piscola/blob/master/LICENSE)
[![Build and Tests](https://github.com/temuller/piscola/actions/workflows/main.yml/badge.svg)](https://github.com/temuller/piscola/actions/workflows/main.yml)
[![Coverage](https://raw.githubusercontent.com/temuller/piscola/master/coverage.svg)](https://raw.githubusercontent.com/temuller/piscola/master/coverage.svg)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/piscola?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/piscola/)
[![ADS -  2022MNRAS.512.3266M ](https://img.shields.io/badge/ADS-_2022MNRAS.512.3266M_-2ea44f)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.3266M/abstract)

Read the full documentation at: [piscola.readthedocs.io](http://piscola.readthedocs.io/). See below for a summary.

___
## Installation

PISCOLA can be installed in the usual ways, via pip:

```
pip install piscola
```

or from source:

```
git clone https://github.com/temuller/piscola.git
cd piscola
pip install .
```

### Requirements

PISCOLA has the following requirements:

```
numpy
pandas
matplotlib
peakutils
requests
sfdmap
extinction
astropy
scipy
george
pickle5
pytest  (optional: for testing the code)
```

### Tests

To run the tests, go to the parent directory and run the following command:

```
pytest -v
```

## Using PISCOLA

PISCOLA can fit the supernova light curves and correct them in a few lines of code:


```python
sn = piscola.call_sn(<sn_file>)
sn.fit()
```

The light-curve parameters are saved in a dictionary and can be accessed directly:

```python
sn.lc_parameters  # dictionary
sn.dm15
```

You can find an example of input file in the [data](https://github.com/temuller/piscola/tree/master/data) directory.

## Citing PISCOLA

If you make use of PISCOLA in your projects, please cite [Müller-Bravo et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.3266M/abstract). See below for the bibtex format:

```code
@ARTICLE{2022MNRAS.512.3266M,
       author = {{M{\"u}ller-Bravo}, Tom{\'a}s E. and {Sullivan}, Mark and {Smith}, Mathew and {Frohmaier}, Chris and {Guti{\'e}rrez}, Claudia P. and {Wiseman}, Philip and {Zontou}, Zoe},
        title = "{PISCOLA: a data-driven transient light-curve fitter}",
      journal = {\mnras},
     keywords = {supernovae: general, cosmology: observations, distance scale, Astrophysics - High Energy Astrophysical Phenomena, Astrophysics - Cosmology and Nongalactic Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2022,
        month = may,
       volume = {512},
       number = {3},
        pages = {3266-3283},
          doi = {10.1093/mnras/stab3065},
archivePrefix = {arXiv},
       eprint = {2110.11340},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.3266M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Contributing and raising an issue

The recommended way is to use the [issues](https://github.com/temuller/piscola/issues) page or send a pull request. Otherwise, you can contact me directly.
