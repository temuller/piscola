# PISCOLA: Python for Intelligent Supernova-COsmology Light-curve Analysis

**Type Ia Supernova Light-curve fitting code in python**


[![repo](https://img.shields.io/badge/GitHub-temuller%2Fpiscola-blue.svg?style=flat)](https://github.com/temuller/piscola)
[![documentation status](https://readthedocs.org/projects/piscola/badge/?version=latest&style=flat)](https://piscola.readthedocs.io/en/latest/?badge=latest)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/temuller/piscola/blob/master/LICENSE)
[![ci](http://img.shields.io/travis/temuller/piscola/master.svg?style=flat)](https://travis-ci.org/temuller/piscola)
![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue)


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

pip install -r requirements.txt

python setup.py install
```

## SFD dust maps

PISCOLA uses the dust maps from the [sfddata](https://github.com/kbarbary/sfddata/) repository. These can be downloaded and moved into the directory where PISCOLA looks for them by defuault using the ``download_dustmaps.py`` script included in this repository (this script relies on [wget](https://pypi.org/project/wget/)):

```
chmod -x download_dustmaps.py

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

or for a "quick" fit with the default parameters:

```python
sn = piscola.call_sn(<sn_name>)
sn.do_magic()
```

You can find an example of input file in the [data](https://github.com/temuller/piscola/tree/master/data) directory.

## Contributing and raising an issue

The recommended way is to use the [issues](https://github.com/temuller/piscola/issues) page. Otherwise, you can contact me directly.
