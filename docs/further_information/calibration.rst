.. _fi_calibration:

Calibration
===========

Magnitude Systems
#################

Magnitude systems are stored under the :code:`mag_sys` directory. In there, the user can find the :code:`magnitude_systems.txt` file that contains the names of the magnitude systems (first column) and their respective files (second column), which contain the zero-points of these magnitude systems:

.. parsed-literal::

	#magnitude_system  	file
	AB			ab_sys_zps.dat
	BD17			bd17_sys_zps.dat
	VEGA			vega_sys_zps.dat
	AB_B12			ab_b12_sys_zps.dat
	AB_SDSS			ab_sdss_sys_zps.dat
	BD17_JLA 		bd17_jla_sys_zps.dat
	BD17_SNLS3 		bd17_snls3_sys_zps.dat
	BD17_SWOPE		bd17_swope_sys_zps.dat
	CSP_BD17		csp_sys_bd17.dat
	CSP_VEGA		csp_sys_vega.dat

The magnitude system files (e.g. ``bd17_sys_zps.dat``) must contain a line which tells PISCOLA which standard SED to use and the standard magnitudes for each filter. For example:

.. parsed-literal::

	standard_sed: bd_17d4708_stisnic_005.dat

	# Start filters for VERSION = PS1s_CFA1_JRK07_DS16
	Bessell_U 	9.724
	Bessell_B  	9.907
	Bessell_V 	9.464
	Bessell_R 	9.166
	Bessell_I 	8.846

	# Start filters for VERSION = PS1s_CFA3_4SHOOTER2_DS17
	4Shooter2_U  	9.6930 
	4Shooter2_B  	9.87400-0.0345 
	4Shooter2_V  	9.47900-0.0087 
	4Shooter2_R  	9.15500-0.021 
	4Shooter2_I  	8.85100-0.0136 

Everything else in the files is ignored (considered as comments). If you want to add another magnitude system, just follow the same structure.

Standard Stars
##############

Standard stars are found in the :code:`standards` directory (e.g., :code:`alpha_lyr_stis_005.dat`, :code:`bd_17d4708_stisnic_005.dat`). If the user wants to use an AB SED, simply set ``standard_sed: AB`` (or ``ab``). The available standard stars are: :math:`BD +17.4708`, Vega (:math:`\alpha Lyr.`) and AB. The latter is defined as:

.. code:: python

	c = 2.99792458e18  # speed of light in [Å/s]
    	sed_wave = np.arange(1000, 250000, 1)  # in [Å]
    	sed_flux = 3631e-23 * c / sed_wave**2  # in [erg s^-1 cm^-2 Å^-1]
    	
Zero-points
###########

The zero-point for a given magnitude system, e.g. Vega. is calculated as follows:

	:math:`ZP_{Vega} = m_{Vega} + 2.5*log_{10}(f_{Vega})`
	
where :math:`m_{Vega}` is the tabulated standard magnitude of Vega in a given filter (taken from the files mentioned above) and :math:`f_{Vega}` the integrated flux through the same given filter (using the standard star SED).
