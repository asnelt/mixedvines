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
    import itertools
    # Manually construct mixed vine
    dim = 3  # Dimension
    vine = MixedVine(dim)
    # Specify marginals
    vine.set_marginal(0, norm(0, 1))
    vine.set_marginal(1, poisson(5))
    vine.set_marginal(2, gamma(2, 0, 4))
    # Specify pair-copulas
    vine.set_copula(1, 0, GaussianCopula(0.5))
    vine.set_copula(1, 1, FrankCopula(4))
    vine.set_copula(2, 0, ClaytonCopula(5))
    # Calculate probability density function on lattice
    bnds = np.empty((3), dtype=object)
    bnds[0] = [-3, 3]
    bnds[1] = [0, 15]
    bnds[2] = [0.5, 25]
    x0, x1, x2 = np.mgrid[bnds[0][0]:bnds[0][1]:0.05, bnds[1][0]:bnds[1][1],
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
