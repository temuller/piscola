.. _basicexamples:

Basic Example
========================

In here we show how to use PISCOLA in the simplest way possible.

As always, we start by importing the necessary packages.

.. code:: python

	import piscola

PISCOLA uses its own format for a SN file (explained in the advanced implementation below) which is similar to the one used by other light-curve fitting codes. As an example we have *SN 2004eo* in a file called ``sn2004eo.dat`` inside the ``data/`` directory. To import a SN all you need to do is use the ``sn_file`` function which receives two arguments, the SN name (or file name) and the directory where to find the file (``data/`` by default).

.. code:: python

	sn = piscola.sn_file('sn2004eo', directory='data_pantheon/')
	print(sn)
	print(f'Observed bands: {sn.bands}')

.. code:: python

	name = sn2004eo, z = 0.01457, ra = 308.226013, dec = 9.92853
	Observed bands: ['csp_u', 'csp_B', 'csp_g', 'csp_o', 'csp_r', 'csp_i']

The ``sn`` object will contain the SN information, i.e., name, redshift, RA, DEC and light curves. The latter are found in ``sn.data``, a dictionary with the observed bands as **keys**, including the zero point (zp), and magnitude system (mag_sys). **Note that PISCOLA accepts fluxes as input, not magnitudes**.

.. code:: python

	print(sn.data.keys())
	print(sn.data['csp_B'].keys())

.. code:: python

	dict_keys(['csp_u', 'csp_B', 'csp_g', 'csp_o', 'csp_r', 'csp_i'])
	dict_keys(['mjd', 'flux', 'flux_err', 'zp', 'mag_sys'])

We can plot the light curves by calling ``sn.plot_data()``.

.. code:: python

	sn.plot_data()

We can also mask the data according to the signal-to-noise ratio and/or phases wanted by using ``sn.mask_data()``. For the latter, and initial fit to the light curves needs to be done.

To fit the light curves one needs to use ``sn.fit_lcs()``, where the user can decide which kernel to use ('*matern52*' by default). One can also plot the fits afterwards by using ``sn.plot_fits()``. From the light curve fits you will get an initial estimation of the rest-frame B-band peak (plotted as a vertical black dashed line). However, we first need to **normalize** the data. All this does is converting the flux units into physical units (if they are not already in physical units) so PISCOLA can work with it.


.. code:: python

	sn.normalize_data()
	sn.fit_lcs()
	sn.plot_fits()

	print('Initial B-band peak estimation:', sn.tmax)

The next step is not find the *mangling function* which will warp the SED template to match the SN colours at the given epochs. This is done by using ``sn.mangle_sed()`` and giving the minimum and maximum phase with respect to B-band peak estimated in the previous step (*-10 and +20 days* by deaulft, respectively). The kernel used can also be chosen ('*squaredexp*' by default). This process can take up to several minutes depending on several factors, but it usually last less than a minute.

.. code:: python

	sn.mangle_sed()

Next comes the estimation of the light-curves parameters for which we use ``sn.calculate_lc_params()``. This step can also take a while to run as it compares the final estimation of the B-band peak with the initial one. If their difference is larger than a certain *threshold* (specified in the code), the whole mangling process is repeated (internally) until convergence is reached.

.. code:: python

	sn.calculate_lc_params()

Finally, we can check the estimated light-curves parameters and plot the rest-frame B-band or any other Bessell band (restricted by the data coverage).

.. code:: python
	
	sn.display_results()
	print(f't_peak = {sn.tmax} +/- {sn.tmax_err}')

.. code:: python
	
	t_peak = 53278.19 +/- 0.58


Putting it all together...

.. code:: python

	sn = piscola.sn_file('sn2004eo', directory='data_pantheon/')

	sn.normalize_data()
	sn.fit_lcs()
	sn.mangle_sed()
	sn.calculate_lc_params()
	sn.display_results()

Or, if you want a "quick" implementation, following the same steps as above and using the default values, you can use ``sn.do_magic()``.

.. code:: python

	sn = piscola.sn_file('sn2004eo', directory='data_pantheon/')
	sn.do_magic()
	sn.display_results()

