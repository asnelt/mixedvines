Code example
============


To manually construct and visualize a simple mixed vine model:

.. code-block:: python

    from scipy.stats import norm, gamma, poisson
    import numpy as np
    from mixedvines.copula import Copula, GaussianCopula, ClaytonCopula, \
            FrankCopula
    from mixedvines.mixedvine import MixedVine
    import matplotlib.pyplot as plt
    # Manually construct mixed vine
    dim = 3  # Dimension
    vine_type = 'c-vine'  # Canonical vine type
    vine = MixedVine(dim, vine_type)
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
