.. _fi_lightcurves:

Light Curves
============

Filters are defined by their own classes in PISCOLA: :class:`Lightcurve` and :class:`Lightcurves`.

Lightcurve
##########

This class represents a single light curve:

.. code:: python

	import piscola

	sn = piscola.call_sn('03D1au')   
	lightcurve = sn.lcs.Megacam_g
	print(lightcurve)

.. code:: python

	band: Megacam_g, zp: -20.846, mag_sys: AB
	
Each light curve will contain information about the observed data, such as time, fluxes, errors, etc.:

This class represents a single light curve:

.. code:: python

	lightcurve.__dict__

.. code:: python

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
	 
For information about the available functions, check the :ref:`Light Curves Class <lightcurves>`.	 
	 
	 
Lightcurves
############

This is a class that wraps the single light curves into a single class:

.. code:: python

	lightcurves = sn.lcs
	lightcurves.__dict__
	
.. code:: python

	{'bands': ['Megacam_g', 'Megacam_r', 'Megacam_i', 'Megacam_z'],
	 'Megacam_g': band: Megacam_g, zp: -20.846, mag_sys: AB,
	 'Megacam_i': band: Megacam_i, zp: -21.834, mag_sys: AB,
	 'Megacam_r': band: Megacam_r, zp: -21.398, mag_sys: AB,
	 'Megacam_z': band: Megacam_z, zp: -22.144, mag_sys: AB}

For information about the available functions, check the :ref:`Light Curves Class <lightcurves>`.	 
