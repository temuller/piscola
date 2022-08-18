.. _fi_extinction:

Extinction Correction
=====================

PISCOLA only correct for Milky Way dust extinction. For this, the packages `sfdmaps <https://github.com/kbarbary/sfdmap>`_ and `extinction <https://github.com/kbarbary/extinction>`_ are used. The ``sfdmaps`` are downloaded automatically the first time extinction correction is applied (Internet connection is therefore necessary), without the user having to interfere.

The available dust extinction laws are: ``ccm89`` (Cardelli, Clayton & Mathis 1989), ``odonnell94`` (O’Donnell 1994), ``fitzpatrick99`` (Fitzpatrick 1999), ``calzetti00`` (Calzetti 2000) and ``fm07`` (Fitzpatrick & Massa 2007; with :math:`R_V = 3.1`).

For information about the available functions, check the :ref:`Extinction Correction functions <extinction_correction>`.
