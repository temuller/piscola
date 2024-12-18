.. PISCOLA documentation master file, created by
   sphinx-quickstart on Mon Jul  1 13:44:03 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PISCOLA's documentation!
===================================

#.. image:: piscola_logo.png
#   :scale: 50 %
#   :align: center
   
PISCOLA is being actively developed in `a public repository on GitHub
<https://github.com/temuller/piscola>`_ so if you have any trouble, `open an issue
<https://github.com/temuller/piscola/issues>`_ there.

.. image:: https://img.shields.io/badge/GitHub-temuller%2Fpiscola-blue.svg?style=flat
    :target: https://github.com/temuller/piscola
.. image:: https://readthedocs.org/projects/piscola/badge/?version=latest&style=flat
    :target: https://piscola.readthedocs.io/en/latest/?badge=latest
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/temuller/piscola/blob/master/LICENSE
.. image:: https://github.com/temuller/piscola/actions/workflows/main.yml/badge.svg
    :target: https://github.com/temuller/piscola/actions/workflows/main.yml
.. image:: https://img.shields.io/badge/Python-3.8%2B-blue.svg
    :target: https://img.shields.io/badge/Python-3.8%2B-blue.svg   
.. image:: https://img.shields.io/pypi/v/piscola?label=PyPI&logo=pypi&logoColor=white
    :target: https://pypi.org/project/piscola/
.. image:: https://img.shields.io/badge/ADS-_2022MNRAS.512.3266M_-2ea44f.svg
    :target: https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.3266M/abstract


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   setup/installation.rst
   setup/conda_env.rst

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_example.ipynb
   examples/sn_file.rst
   examples/extensive_example.ipynb
   
.. toctree::
   :maxdepth: 1
   :caption: Further Information
   
   further_information/filters.rst
   further_information/lightcurves.rst
   further_information/gaussian_process.rst
   further_information/extinction_correction.rst
   further_information/calibration.rst  
   further_information/supernova.rst

.. toctree::
   :maxdepth: 1
   :caption: API User Guide

   user/sn_class.rst
   user/filters_class.rst
   user/lightcurves_class.rst
   user/gaussian_process.rst
   user/extinction_correction.rst
   user/utils.rst
   
.. toctree::
   :maxdepth: 1
   :caption: About the Code

   about/details.rst
   
   
Citing PISCOLA
--------------

If you make use of PISCOLA, please cite the following `paper <https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.3266M/abstract>`_:

.. code-block:: tex

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


License & Attribution
---------------------

Copyright 2021 Tomás E. Müller Bravo.

PISCOLA is being developed by `Tomás E. Müller Bravo <https://temuller.github.io/>`_ in a
`public GitHub repository <https://github.com/temuller/piscola>`_.
The source code is made available under the terms of the MIT license.
