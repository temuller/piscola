# PISCOLA: Python for Intelligent Supernova-COsmology Light-curve Analysis

**Supernova light-curve fitting code in python**


[![repo](https://img.shields.io/badge/GitHub-temuller%2Fpiscola-blue.svg?style=flat)](https://github.com/temuller/piscola)
[![documentation status](https://readthedocs.org/projects/piscola/badge/?version=latest&style=flat)](https://piscola.readthedocs.io/en/latest/?badge=latest)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/temuller/piscola/blob/master/LICENSE)
[![Build Status](https://app.travis-ci.com/temuller/piscola.svg?branch=master)](https://app.travis-ci.com/temuller/piscola)
![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/piscola?label=PyPI&logo=pypi&logoColor=white)](https://pypi.org/project/piscola/)
[![Conda Version](https://img.shields.io/conda/vn/temuller/piscola?label=conda%20version)](https://anaconda.org/temuller/piscola)

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

## SFD dust maps

PISCOLA uses the dust maps from the [sfddata](https://github.com/kbarbary/sfddata/) repository. These can be downloaded and moved into the directory where PISCOLA looks for them by default, by using the ``download_dustmaps.py`` script included in this repository (this script relies on [wget](https://pypi.org/project/wget/)):

```
python download_dustmaps.py piscola
```

## Recommended installation

Here is an easy way of installing and making PISCOLA work:

```
conda create -n pisco pip  # creates an environment called pisco with pip
conda activate pisco
pip install piscola
wget https://raw.githubusercontent.com/temuller/piscola/master/download_dustmaps.py
python download_dustmaps.py piscola
```

## Using PISCOLA

PISCOLA can fit the supernova light curves and correct them in a few lines of code:


```python
sn = piscola.call_sn(<sn_name>)

sn.normalize_data()
sn.fit_lcs()
sn.mangle_sed()
sn.calculate_lc_params()
```

or if you are OK with using the default parameters, [you can do magic](https://www.youtube.com/watch?v=tt4cR9szMS8):

```python
sn = piscola.call_sn(<sn_name>)
sn.do_magic()
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

The recommended way is to use the [issues](https://github.com/temuller/piscola/issues) page. Otherwise, you can contact me directly.
