=============================
mixedvines Package for Python
=============================

Package for canonical vine copula trees with mixed continuous and discrete
marginals.  If you use this software for publication, please cite [ONKEN2016]_.


Description
-----------

This packge contains a complete framework based on canonical vine copulas for
modelling multivariate data that are partly discrete and partly continuous.
The resulting multivariate distributions are flexible with rich dependence
structures and marginals.

For continuous marginals, implementations of the normal and the gamma
distributions are provided.  For discrete marginals, Poisson, binomial and
negative binomial distributions are provided.  As bivariate copula building
blocks, the Gaussian, Frank and Clayton families as well as rotation transformed
families are provided.  Additional marginal and pair copula distributions can be
added easily.

The package includes methods for sampling, likelihood calculation and inference,
all of which have quadratic complexity.  These procedures are combined to
estimate entropy by means of Monte Carlo integration.

Please see [ONKEN2016]_ for a more detailed description of the framework.


Prerequisites
-------------

The package is compatible with Python 2 and 3 and additionaly only requires
`SciPy
<http://www.scipy.org/install.html>`_.


Usage
-----

Suppose that data are given in a NumPy array ``samples`` with shape ``(n, d)``,
where ``n`` is the number of samples and ``d`` is the number of elements.
First, specify which of the elements are continuous.  If, for instance, the
distribution has three elements and the first and last element are continuous
whereas the second element is discrete:

    import numpy as np
    is_continuous = np.full((3), True, dtype=bool)
    is_continuous[1] = False

To fit a mixed vine to the samples:

    from mixedvine import MixedVine
    vine = MixedVine.fit(samples, is_continuous)

``vine`` is now a ``MixedVine`` object.  To draw samples from the distribution,
calculate their density and estimate the distribution entropy in units of bits:

    samples = vine.rvs(size=100)
    logpdf = vine.logpdf(samples)
    (entropy, standard_error_mean) = vine.entropy(sem_tol=1e-2)

Note that for the canonical vine, the order of elements is important.  Elements
should be sorted according to the importance of their dependencies to other
elements where elements with important dependencies to many other elements should
come first.  A heuristic way to select the order of elements is to calculate
Kendall's tau between all element pairs (see ``scipy.stats.kendalltau``), to
obtain a score for each element by summing the tau's of the pairs the element
occurs in and to sort elements in descending order according to their scores.


References
----------

.. [ONKEN2016] A. Onken and S. Panzeri (2016). Mixed vine copulas as joint models
   of spike counts and local field potentials.  In D. D. Lee, M. Sugiyama,
   U. V. Luxburg, I. Guyon and R. Garnett, editors, Advances in Neural
   Information Processing Systems 29 (NIPS 2016), pages 1325-1333.


License
-------

Copyright (C) 2017 Arno Onken

This file is part of the mixedvines package.

The mixedvines package is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or (at your option) any
later version.

The mixedvines package is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program; if not, see <http://www.gnu.org/licenses/>.
