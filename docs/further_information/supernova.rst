.. _fi_supernova:

Understanding PISCOLA
=====================

In this section, the main class of PISCOLA is described, and the entire fitting and correction processes are explained. Please, read carefully to understand how PISCOLA works. If any inconsistencies are found, please notify me, open an `issue <https://github.com/temuller/piscola/issues>`_ or send a pull request.

The Supernova Class
###################

The :class:`Supernova` class is the main class in PISCOLA, it represents a supernova (SN) and incorporates all the other classes described in the other sections (i.e. filters, light curves and SED). The :class:`Supernova` object is not called directly, but through the :func:`call_sn()` function: 

.. code:: python

	import piscola

	sn = piscola.call_sn('03D1au.dat')   

The filters, light curves and SED objects can be found in ``sn.filters``, ``sn.lcs`` and ``sn.sed``, respectively. The following parameters can be initially found in the :class:`Supernova` object:


.. code:: python

	sn.__dict__
	
.. code:: python

	{'name': '03D1au',
	 'z': 0.50349,
	 'ra': 36.043209000000004,
	 'dec': -4.037469000000001,
	 'init_fits': {},
	 'init_lcs': ['Megacam_g' 'Megacam_i' 'Megacam_r' 'Megacam_z'],
	 'lcs': ['Megacam_g', 'Megacam_r', 'Megacam_i', 'Megacam_z'],
	 'filters': ['Bessell_U', 'Bessell_B', 'Megacam_g', 'Bessell_V', 'Megacam_r', 'Bessell_R', 'Megacam_i', 'Bessell_I', 'Megacam_z'],
	 'bands': ['Megacam_g', 'Megacam_r', 'Megacam_i', 'Megacam_z'],
	 'sed': name: conley09f, z: 0.50349, ra: 36.043209000000004, dec: -4.037469000000001}



Light Curves Fitting and Correction
###################################

The user can simply fit and correct the SN's light curves with :func:`sn.fit()`. However, there is a lot happening behind this "simple" function.


Light Curves
************

The light-curve data found in the SN file (e.g. ``03D1au.dat``) does not need to have any units in particular. It is common practice that the surveys use a single zero-point (ZP) for all the filters. This means that the flux will have some arbitrary units, but the values of the magnitude are kept unchanged. The initial light-curve data is kept in ``sn.init_lcs``. The magnitudes and errors are automatically calculates by PISCOLA with the following equations:

	:math:`m = -2.5*log_{10}(F) + ZP`
	
	:math:`\sigma_{m} = \frac{2.5*\sigma_f}{f*ln(10)}`
	
where :math:`f` is the flux, :math:`\sigma_f` is the flux error, :math:`ZP`, :math:`m` is the magnitude, :math:`\sigma_m` is the magnitude error is the zero-point and :math:`ln()` is the natural logarithm (base :math:`e`). The convertion from magnitudes to flux has the following equations:

	:math:`f = 10^{-0.4*(m - ZP)}`
	
	:math:`\sigma_{f} = | 0.4*f*ln(10)*\sigma_m|`

The data in ``sn.lcs`` contains the same information as in ``sn.init_lcs``, but re-normalise, and these are the ones used by PISCOLA. What this re-normalisation does it to convert the ZPs and fluxes using the magnitude system provided (e.g. ``sn.lcs.Megacam_g.mag_sys``). If the data is, for example, in the ``AB`` magnitude system, the appropiate ZPs are used and the flux units are converted accordingly following the equation below:

	:math:`f_{AB} = f*10^{-0.4*(ZP - ZP_{AB})}`

where :math:`f` is the flux in the initial units and :math:`f_{AB}` is the flux in physical units (:math:`erg\,s^{-1} cm^{-2} Å^{-1}`). Note that the values of the magnitudes do not change, just those of the fluxes and ZPs. This convertion is important as the different scales between different bands matter for the light-curve fits (described below). For more information about the magnitude systems, check the :ref:`Calibration Section<fi_calibration>`.


Gaussian Process Fits
*********************

The light curves are initially fit with Gaussian Process (GP) to obtain an estimation of the epoch of :math:`B`-band peak mangitude (using `peakutils <https://peakutils.readthedocs.io/en/latest/>`_). This fit is in 2D, i.e. flux (plus errors) as a function of time and wavelength at the same time. In the time axis, a `Mátern-5/2` kernel is used by default as it is effective at describing the light curves of type Ia SNe (SNe Ia). For the wavelength axis, a `Squared Exponential` kernel is used as the "distances" between bands can be large (specially in the near-infrared). The light curves are centred at their respective effective wavelengths, which are calculated using the SED template at phase equal zero.

Correction Process
******************

The photons of a SN go through several processes since they are emitted until they reach an observer. These are the following:

	1) photons are emitted by the SN, get extincted by circumstellar dust and dust from the host galaxy;
	2) they travel to our Milky Way galaxy, getting **redshifted** and extincted by intergalactic dust on the way;
	3) the photons that reach our galaxy are **extincted by Milky Way dust**;
	4) and finally, the photons reach an observer.
	
PISCOLA replicates the processes marked in bold to correct the SN light curves. The circumstellar, host-galaxy and intergalactic dust extinctions are omitted as these are not easy to estimate.
The steps that PISCOLA takes are the following: 

	1) the SED template is **redshifted**;
	2) **dust extinction** is **applied** to it;
	3) the SED is then **mangled** to match the observed colours of the SN (this is described below);
	4) it then gets **corrected** for **dust extinction**;
	5) and finally, the SED template gets **blueshifted** (i.e. de-redshifted).
	
This produces an SED template that is an approximation of the real SED of the SN, assuming that the SED template is a good initial guess. From the corrected SED, restframe light curves and light-curve parameters are calculated: peak magnitude in the :math:`B` band (``Bmax``) and its epoch (``tmax``), :math:`\Delta m_{15}(B)` (``dm15``; as defined in `Phillips 1993 <https://ui.adsabs.harvard.edu/abs/1993ApJ...413L.105P/abstract>`_), :math:`(B-V)_{max}` (``colour``; at the epoch of peak magnitude in the :math:`B` band) and :math:`s_{BV}` (``sBV``; as defined in `Burns et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014ApJ...789...32B/abstract>`_).


Mangling
********

The SED template is `mangled` to match the observed colours of the SN. For this, one must compred the observed light curves vs those from the template (therefore, the re-normalisation of the light curves described above is needed). The flux ratio between both (:math:`f_{obs}/f_{temp}`) provides the necessary information to modify the template to match the observations. GP is used to fit in 2D this ratios surface, producing a `magnling surface` (a.k.a. the `mangling function`), which is then convolved with the SED template. The actual fits to the observed light curves are produced out of this mangled SED, which is also used in the correction process, as mentioned above.

The advantage of fitting the mangling surface instead of directly fitting the light curves is that, if the initial SED template is a good representation of the actual SED of the SN, the surface should be approximately flat. This means that the fits are relatively simple and straight forward. Furthermore, one does not need to worry about differences in scale between different bands. Finally, the 2D fits provides an informative interpolation/extrapolation.
