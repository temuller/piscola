.. _basicexamples:

Basic Example
========================

Simple Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In here, we show how to use PISCOLA in the simplest way possible. As always, start by importing the necessary packages.

.. code:: python

	import piscola
	version = piscola.__version__
	print(f'PISCOLA version: v{version}')
	
.. code:: python

	PISCOLA version: v1.0.0
	

PISCOLA uses its own format for a SN file (explained SOMEWHERE ELSE) which has a similar format to that used by other codes. As an example, we have SN ``03D1au`` (from the SNLS survey) in a file called ``03D1au.dat``. This file can be downloaded from the `repository <https://github.com/temuller/piscola/tree/master/data>`_. To import a SN, all that needs to be done is use the :func:`call_sn()` function, which receives the name of the file as an argument:

.. code:: python

	sn = piscola.call_sn('03D1au.dat')
	print(sn)
	print(f'Observed bands: {sn.bands}')

.. code:: python

	name: 03D1au, z: 0.50349, ra: 36.043, dec: -4.0375
	Observed bands: ['Megacam_g', 'Megacam_r', 'Megacam_i', 'Megacam_z']

The ``sn`` object will contain all the necessary information, i.e. name, redshift, RA, DEC and the observed multi-colour light curves. The latter are found in ``sn.lcs``, a :func:`Lightcurves` object, which also includes the zero-points (``zp``), and magnitude system (``mag_sys``):

.. code:: python

	print(sn.lcs)
	print(sn.lcs.Megacam_g)
	sn.lcs.Megacam_g.__dict__

.. code:: python

	['Megacam_g', 'Megacam_r', 'Megacam_i', 'Megacam_z']
	band: Megacam_g, zp: -20.846, mag_sys: AB

	{'band': 'Megacam_g',
	 'time': array([52880.58, 52900.49, 52904.6 , 52908.53, 52930.39, 52934.53,
		52937.55, 52944.39, 52961.45, 52964.37, 52992.33, 52999.32]),
	 'flux': array([-1.85848101e-20,  1.70044129e-18,  1.89317266e-18,  1.85866457e-18,
		 4.68383103e-19,  3.39987304e-19,  3.07085307e-19,  1.45787510e-19,
		 1.58865710e-19,  8.00752930e-20,  8.87940928e-20,  2.56975152e-21]),
	 'flux_err': array([3.93722644e-20, 9.87059915e-20, 5.70393061e-20, 5.37353399e-20,
		4.80910642e-20, 4.34563338e-20, 7.37426910e-20, 8.37463666e-20,
		7.89280825e-20, 5.21292452e-20, 6.20411439e-20, 5.51119925e-20]),
	 'zp': -20.845742237479524,
	 'mag': array([        nan, 23.57785366, 23.4612822 , 23.48125521, 24.97775471,
		25.32560101, 25.43611017, 26.24495696, 26.15168234, 26.89551142,
		26.78329758, 30.62952993]),
	 'mag_err': array([        nan,  0.06302403,  0.03271209,  0.03138942,  0.11147757,
		 0.13877611,  0.26072595,  0.62369171,  0.53941833,  0.70681738,
		 0.75861258, 23.28516396]),
	 'mag_sys': 'AB'}

The light curves can be plotted by calling the function :func:`sn.plot_lcs()`:

.. code:: python

	sn.plot_lcs()

.. image:: basic_example/03D1au_lcs.png

To fit the light curves one needs to use :func:`sn.fit()`, where the user can decide which kernels to use. By default, PISCOLA uses ``matern52`` for the `time` axis, and ``squaredexp`` for the `wavelength` axis. One can also plot the fits afterwards by using ``sn.plot_fits()``. From the fits, one gets an estimation of the epoch of rest-frame B-band peak (plotted as a vertical dashed line):


.. code:: python

	sn.fit()
	sn.plot_fits()

.. image:: basic_example/03D1au_lc_fits.png


The fitting process includes: the fits of the observed light curves, Milky-Way extinction correction and mangling of the SED template. Finally, we can check the calculated light-curves parameters:

.. code:: python
	
	sn.lc_parameters

.. code:: python
	
	{'tmax': 52907.9,
	 'tmax_err': 0.934,
	 'Bmax': 23.093,
	 'Bmax_err': 0.008,
	 'dm15': 0.825,
	 'dm15_err': 0.017,
	 'colour': 0.07,
	 'colour_err': 0.017,
	 'sBV': 0.967,
	 'sBV_err': 0.033}

``Nan`` values correspond to parameters that were not calculated due limited data coverage.

In Summary
~~~~~~~~~~

Simply follow these steps:

.. code:: python
	
	import piscola
	
	sn = piscola.call_sn('03D1au.dat')
	sn.fit()
