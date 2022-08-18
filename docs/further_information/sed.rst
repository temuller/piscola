.. _fi_sed:

Spectral Energy Distribution: templates and more
================================================

The SED Object
##############

The spectral energy distribution (SED) template that PISCOLA uses to correct the light curves is represented by the :class:`SEDTemplate` class and can be called in the following way:


.. code:: python

	from piscola.sed_class import SEDTemplate

	sed = SEDTemplate()
	print(sed)
	
.. code:: python

	name: conley09f, z: 0.0, ra: None, dec: None

The ``name`` refers to the name of the SED template in the ``templates`` directory in PISCOLA (``conley09f`` by default). The ``sed`` object can also be initiallised with different coordinates (``ra`` and ``dec``) and redshift (``z``) which are used for correcting the light curves from dust extinction and time dilation. This object contains the information shown below:

	
.. code:: python

	sed.__dict__
	
.. code:: python
	
	{'z': 0.0,
	 'ra': None,
	 'dec': None,
	 'data':         phase     wave          flux
	 0       -20.0   1000.0  3.533845e-38
	 1       -20.0   1010.0  3.533845e-38
	 2       -20.0   1020.0  3.533845e-38
	 3       -20.0   1030.0  3.533845e-38
	 4       -20.0   1040.0  3.533845e-38
	 ...       ...      ...           ...
	 243160   85.0  23890.0  5.169887e-12
	 243161   85.0  23900.0  5.223790e-12
	 243162   85.0  23910.0  5.278787e-12
	 243163   85.0  23920.0  5.334638e-12
	 243164   85.0  23930.0  5.391089e-12
	 
	 [243165 rows x 3 columns],
	 'phase': array([-20., -20., -20., ...,  85.,  85.,  85.]),
	 'wave': array([ 1000.,  1010.,  1020., ..., 23910., 23920., 23930.]),
	 'flux': array([3.5338449e-38, 3.5338449e-38, 3.5338449e-38, ..., 5.2787867e-12,
		5.3346382e-12, 5.3910892e-12]),
	 'flux_err': array([0., 0., 0., ..., 0., 0., 0.]),
	 'name': 'conley09f',
	 'comments': '',
	 'redshifted': False,
	 'extincted': False}
	
where ``data`` is a pandas DataFrame. If a ``README.txt`` file is found in the ``templates`` directory, it is added to the ``comments``. 

The available SED templates can be shown in the following way:

.. code:: python

	sed.show_available_templates()
	
.. code:: python
	
	List of available SED templates: ['conley09f', 'guy07', 'jla']

For information about the available functions, check the :ref:`SED Class<sed>`.

Adding New SED Templates
########################

SEDs are stored under the :code:`templates` directory, in their own respective directories. PISCOLA reads the file :code:`sed_template.dat` to import a template, where the first column of the file are the phases in units of days (e.g. epochs with respect to :math:`B`-band peak magnitude), the second one are the wavelengths in angstroms, and the last one are the fluxes in any arbitrary units (as a function of wavelength). If you want to add another template, just follow the same structure.

