=============================
mixedvines Package for Python
=============================

Package for canonical vine copula trees with mixed continuous and discrete
marginals.  If you use this software for publication, please cite [ONKEN2016]_.


Description
-----------

This package contains a complete framework based on canonical vine copulas for
modelling multivariate data that are partly discrete and partly continuous.  The
resulting multivariate distributions are flexible with rich dependence
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


Documentation
-------------

The full documentation for the mixedvines package is available at
`Read the Docs
<http://mixedvines.readthedocs.io/>`_.


Requirements
------------

The package is compatible with Python 2.7 and 3.x and additionaly requires
`NumPy and SciPy
<http://www.scipy.org/install.html>`_.


Installation
------------

To install the mixedvines package, run::

    pip install mixedvines


Usage
-----

Suppose that data are given in a NumPy array ``samples`` with shape ``(n, d)``,
where ``n`` is the number of samples and ``d`` is the number of elements per
sample.  First, specify which of the elements are continuous.  If, for instance,
the distribution has three elements and the first and last element are
continuous whereas the second element is discrete:

.. code-block:: python

    is_continuous = [True, False, True]

To fit a mixed vine to the samples:

.. code-block:: python

    from mixedvines.mixedvine import MixedVine
    vine = MixedVine.fit(samples, is_continuous)

``vine`` is now a ``MixedVine`` object.  To draw samples from the distribution,
calculate their density and estimate the distribution entropy in units of bits:

.. code-block:: python

    samples = vine.rvs(size=100)
    logpdf = vine.logpdf(samples)
    (entropy, standard_error_mean) = vine.entropy(sem_tol=1e-2)

To manually construct and visualize a simple mixed vine model:

.. code-block:: python

    from scipy.stats import norm, gamma, poisson
    import numpy as np
    from mixedvines.copula import GaussianCopula, ClaytonCopula, FrankCopula
    from mixedvines.mixedvine import MixedVine
    import matplotlib.pyplot as plt
    import itertools
    # Manually construct mixed vine
    dim = 3  # Dimension
    vine = MixedVine(dim)
    # Specify marginals
    vine.set_marginal(0, norm(0, 1))
    vine.set_marginal(1, poisson(5))
    vine.set_marginal(2, gamma(2, 0, 4))
    # Specify pair copulas
    vine.set_copula(1, 0, GaussianCopula(0.5))
    vine.set_copula(1, 1, FrankCopula(4))
    vine.set_copula(2, 0, ClaytonCopula(5))
    # Calculate probability density function on lattice
    bnds = np.empty((3), dtype=object)
    bnds[0] = [-3, 3]
    bnds[1] = [0, 15]
    bnds[2] = [0.5, 25]
    (x0, x1, x2) = np.mgrid[bnds[0][0]:bnds[0][1]:0.05, bnds[1][0]:bnds[1][1],
                            bnds[2][0]:bnds[2][1]:0.1]
    points = np.array([x0.ravel(), x1.ravel(), x2.ravel()]).T
    pdf = vine.pdf(points)
    pdf = np.reshape(pdf, x1.shape)
    # Generate random variates
    size = 100
    samples = vine.rvs(size)
    # Visualize 2d marginals and samples
    comb = list(itertools.combinations(range(dim), 2))
    for i, cmb in enumerate(comb):
        # Sum over all axes not in cmb
        cmb_inv = tuple(set(range(dim)) - set(cmb))
        margin = np.sum(pdf, axis=cmb_inv).T
        plt.subplot(2, len(comb), i + 1)
        plt.imshow(margin, aspect='auto', interpolation='none', cmap='hot',
                   origin='lower', extent=[bnds[cmb[0]][0], bnds[cmb[0]][1],
                                           bnds[cmb[1]][0], bnds[cmb[1]][1]])
        plt.ylabel('$x_' + str(cmb[1]) + '$')
        plt.subplot(2, len(comb), len(comb) + i + 1)
        plt.scatter(samples[:, cmb[0]], samples[:, cmb[1]], s=1)
        plt.xlim(bnds[cmb[0]][0], bnds[cmb[0]][1])
        plt.ylim(bnds[cmb[1]][0], bnds[cmb[1]][1])
        plt.xlabel('$x_' + str(cmb[0]) + '$')
        plt.ylabel('$x_' + str(cmb[1]) + '$')
    plt.tight_layout()
    plt.show()

This code shows the 2d marginals and 100 samples of a 3d mixed vine.


Source code
-----------

The source code of the mixedvines package is hosted on
`GitHub
<https://github.com/asnelt/mixedvines/>`_.


References
----------

.. [ONKEN2016] A. Onken and S. Panzeri (2016).  Mixed vine copulas as joint
   models of spike counts and local field potentials.  In D. D. Lee,
   M. Sugiyama, U. V. Luxburg, I. Guyon and R. Garnett, editors, Advances in
   Neural Information Processing Systems 29 (NIPS 2016), pages 1325-1333.


License
-------

Copyright (C) 2017-2019 Arno Onken

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
