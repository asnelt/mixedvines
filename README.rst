=============================
mixedvines Package for Python
=============================

Package for canonical vine copula trees with mixed continuous and discrete
marginals.  If you use this software for publication, please cite [ONKEN2016]_.


Description
-----------

This packge contains a complete framework based on canonical vine copulas for
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

.. code-block:: python

    import numpy as np
    is_continuous = np.full((3), True, dtype=bool)
    is_continuous[1] = False

To fit a mixed vine to the samples:

.. code-block:: python

    from mixedvine import MixedVine
    vine = MixedVine.fit(samples, is_continuous)

``vine`` is now a ``MixedVine`` object.  To draw samples from the distribution,
calculate their density and estimate the distribution entropy in units of bits:

.. code-block:: python

    samples = vine.rvs(size=100)
    logpdf = vine.logpdf(samples)
    (entropy, standard_error_mean) = vine.entropy(sem_tol=1e-2)

Note that for the canonical vine, the order of elements is important.  Elements
should be sorted according to the importance of their dependencies to other
elements where elements with important dependencies to many other elements
should come first.  A heuristic way to select the order of elements is to
calculate Kendall's tau between all element pairs
(see ``scipy.stats.kendalltau``), to obtain a score for each element by summing
the tau's of the pairs the element occurs in and to sort elements in descending
order according to their scores.

To manually construct and visualize a simple mixed vine model:

.. code-block:: python

    from scipy.stats import norm, gamma, poisson
    import numpy as np
    from mixedvines.copula import Copula, GaussianCopula, ClaytonCopula, \
            FrankCopula
    from mixedvines.mixedvine import Marginal, MixedVine
    import matplotlib.pyplot as plt
    # Manually construct mixed vine
    dim = 3  # Dimension
    vine_type = 'c-vine'  # Canonical vine type
    vine = MixedVine(dim, vine_type)
    # Specify marginals
    marginals = np.empty(dim, dtype=Marginal)
    marginals[0] = Marginal(norm(0, 1))
    marginals[1] = Marginal(poisson(5))
    marginals[2] = Marginal(gamma(2, 0, 4))
    # Specify pair copulas
    copulas = np.empty((dim - 1, dim), dtype=Copula)
    copulas[0, 0] = GaussianCopula(0.5)
    copulas[0, 1] = FrankCopula(4)
    copulas[1, 0] = ClaytonCopula(5)
    # Set marginals and pair copulas
    for marginal_index, marginal in enumerate(marginals):
        vine.set_marginal(marginal, marginal_index)
    for layer_index in range(1, dim):
        for copula_index in range(dim - layer_index):
            vine.set_copula(copulas[layer_index - 1, copula_index],
                            copula_index, layer_index)
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
    m01 = np.sum(pdf, axis=2).T
    m02 = np.sum(pdf, axis=1).T
    m12 = np.sum(pdf, axis=0).T
    plt.subplot(2, 3, 1)
    plt.imshow(m01, aspect='auto', interpolation='none', cmap='hot',
               origin='lower', extent=[bnds[0][0], bnds[0][1], bnds[1][0],
               bnds[1][1]])
    plt.ylabel('$x_1$')
    plt.subplot(2, 3, 2)
    plt.imshow(m02, aspect='auto', interpolation='none', cmap='hot',
               origin='lower', extent=[bnds[0][0], bnds[0][1], bnds[2][0],
               bnds[2][1]])
    plt.ylabel('$x_2$')
    plt.subplot(2, 3, 3)
    plt.imshow(m12, aspect='auto', interpolation='none', cmap='hot',
               origin='lower', extent=[bnds[1][0], bnds[1][1], bnds[2][0],
               bnds[2][1]])
    plt.ylabel('$x_2$')
    # Plot samples
    plt.subplot(2, 3, 4)
    plt.scatter(samples[:, 0], samples[:, 1], s=1)
    plt.xlim(bnds[0][0], bnds[0][1])
    plt.ylim(bnds[1][0], bnds[1][1])
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.subplot(2, 3, 5)
    plt.scatter(samples[:, 0], samples[:, 2], s=1)
    plt.xlim(bnds[0][0], bnds[0][1])
    plt.ylim(bnds[2][0], bnds[2][1])
    plt.xlabel('$x_0$')
    plt.ylabel('$x_2$')
    plt.subplot(2, 3, 6)
    plt.scatter(samples[:, 1], samples[:, 2], s=1)
    plt.xlim(bnds[1][0], bnds[1][1])
    plt.ylim(bnds[2][0], bnds[2][1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.tight_layout()
    plt.show()

This code shows the 2d marginals and 100 samples of a 3d mixed vine.


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
