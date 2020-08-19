# Python for Intelligent Supernova-COsmology Light-Curve Analysis: PISCOLA

**Type Ia Supernova Light-curve fitting code in python**


[![repo](https://img.shields.io/badge/GitHub-temuller%2Fpiscola-blue.svg?style=flat)](https://github.com/temuller/piscola)
[![license](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/temuller/piscola/blob/master/LICENSE)
[![ci](http://img.shields.io/travis/temuller/piscola/master.svg?style=flat)](https://travis-ci.org/temuller/piscola)


Read the documentation at: [piscola.readthedocs.io](http://piscola.readthedocs.io/).

Dependencies:
  - python 3.6+
  - numpy
  - matplotlib
  - pandas
  - lmfit
  - peakutils
  - george
  - emcee
  - extinction
  - sfdmap
  - astropy
  
It is recommended to create a conda environment:

	`conda config --add channels conda-forge`

	`conda create -n pisco numpy matplotlib pandas lmfit peakutils george emcee extinction sfdmap astropy`

Otherwise, you can install the dependencies by typing `pip install -r requirements.txt` on your terminal.


In order to use PISCOLA in any directory, you will need to clone the repository and install it by typing `python setup.py install` inside the directory.
