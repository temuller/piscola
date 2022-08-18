.. _fi_gaussian_process:

Gaussian Process: light-curve fits
==================================

PISCOLA uses `george <https://george.readthedocs.io/en/latest/>`_ to fit the multi-colour light curves. These are fit in 2D, i.e. `flux` (plus `errors`) as a function of `time` and `wavelength` at the same time. This allows PISCOLA to do a more informative interpolation and extrpolation.

The available kernels are: ``matern32`` (Mátern 3/2), ``matern52`` (Mátern 5/2) and ``squaredexp`` (Squared-Exponential).

For information about the available functions, check the :ref:`Gaussian Process functions <gaussian_process>`.
