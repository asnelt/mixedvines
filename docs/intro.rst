Introduction
============


This package contains a complete framework based on canonical vine copulas for
modeling multivariate data that are partly discrete and partly continuous.
The resulting multivariate distributions are flexible with rich dependence
structures and marginals.

For continuous marginals, implementations of the normal and the gamma
distributions are provided.  For discrete marginals, Poisson, binomial and
negative binomial distributions are provided.  As bivariate copula building
blocks, the Gaussian, Frank and Clayton families as well as rotation
transformed families are provided.  Additional marginal and pair-copula
distributions can be added easily.

The package includes methods for sampling, likelihood calculation and
inference, all of which have quadratic complexity.  These procedures are
combined to estimate entropy by means of Monte Carlo integration.
