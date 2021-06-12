# -*- coding: utf-8 -*-
# Copyright (C) 2017-2019, 2021 Arno Onken
#
# This file is part of the mixedvines package.
#
# The mixedvines package is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# The mixedvines package is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
'''
This module implements univariate marginal distribution that are either
continuous or discrete.
'''
from scipy.stats import rv_continuous, norm, gamma, poisson, binom, nbinom
import numpy as np


class Marginal(object):
    '''
    This class represents a marginal distribution which can be continuous or
    discrete.

    Parameters
    ----------
    rv_mixed : scipy.stats.distributions.rv_frozen
        The distribution object, either of a continuous or of a discrete
        univariate distribution.

    Attributes
    ----------
    rv_mixed : scipy.stats.distributions.rv_frozen
        The distribution object.
    is_continuous : boolean
        `True` if the distribution is continuous.

    Methods
    -------
    logpdf(samples)
        Log of the probability density function or probability mass function.
    pdf(samples)
        Probability density function or probability mass function.
    logcdf(samples)
        Log of the cumulative distribution function.
    cdf(samples)
        Cumulative distribution function.
    ppf(samples)
        Inverse of the cumulative distribution function.
    rvs(size, random_state)
        Generate random variates.
    fit(samples, is_continuous)
        Fit a distribution to samples.
    '''

    def __init__(self, rv_mixed):
        self.rv_mixed = rv_mixed
        self.is_continuous = isinstance(rv_mixed.dist, rv_continuous)

    def logpdf(self, samples):
        '''
        Calculates the log of the probability density function.

        Parameters
        ----------
        samples : array_like
            Array of samples.

        Returns
        -------
        ndarray
            Log of the probability density function evaluated at `samples`.
        '''
        if self.is_continuous:
            return self.rv_mixed.logpdf(samples)
        return self.rv_mixed.logpmf(samples)

    def pdf(self, samples):
        '''
        Calculates the probability density function.

        Parameters
        ----------
        samples : array_like
            Array of samples.

        Returns
        -------
        ndarray
            Probability density function evaluated at `samples`.
        '''
        return np.exp(self.logpdf(samples))

    def logcdf(self, samples):
        '''
        Calculates the log of the cumulative distribution function.

        Parameters
        ----------
        samples : array_like
            Array of samples.

        Returns
        -------
        ndarray
            Log of the cumulative distribution function evaluated at `samples`.
        '''
        return self.rv_mixed.logcdf(samples)

    def cdf(self, samples):
        '''
        Calculates the cumulative distribution function.

        Parameters
        ----------
        samples : array_like
            Array of samples.

        Returns
        -------
        ndarray
            Cumulative distribution function evaluated at `samples`.
        '''
        return np.exp(self.logcdf(samples))

    def ppf(self, samples):
        '''
        Calculates the inverse of the cumulative distribution function.

        Parameters
        ----------
        samples : array_like
            Array of samples.

        Returns
        -------
        ndarray
            Inverse of the cumulative distribution function evaluated at
            `samples`.
        '''
        return self.rv_mixed.ppf(samples)

    def rvs(self, size=1, random_state=None):
        '''
        Generates random variates from the distribution.

        Parameters
        ----------
        size : int, optional
            The number of samples to generate.  (Default: 1)
        random_state : {None, int, RandomState, Generator}, optional
            The random state to use for random variate generation.  `None`
            corresponds to the `RandomState` singleton.  For an int, a new
            `RandomState` is generated and seeded.  For a `RandomState` or
            `Generator`, the object is used.  (Default: `None`)

        Returns
        -------
        array_like
            Array of samples.
        '''
        return self.rv_mixed.rvs(size, random_state=random_state)

    @staticmethod
    def fit(samples, is_continuous):
        '''
        Fits a distribution to the given samples.

        Parameters
        ----------
        samples : array_like
            Array of samples.
        is_continuous : bool
            If `True` then a continuous distribution is fitted.  Otherwise, a
            discrete distribution is fitted.

        Returns
        -------
        best_marginal : Marginal
            The distribution fitted to `samples`.
        '''
        # Mean and variance
        mean = np.mean(samples)
        var = np.var(samples)
        # Set suitable distributions
        if is_continuous:
            if np.any(samples <= 0):
                options = [norm]
            else:
                options = [norm, gamma]
        else:
            if var > mean:
                options = [poisson, binom, nbinom]
            else:
                options = [poisson, binom]
        params = np.empty(len(options), dtype=object)
        marginals = np.empty(len(options), dtype=object)
        # Fit parameters and construct marginals
        for i, dist in enumerate(options):
            if dist == poisson:
                params[i] = [mean]
            elif dist == binom:
                param_n = np.max(samples)
                param_p = np.sum(samples) / (param_n * len(samples))
                params[i] = [param_n, param_p]
            elif dist == nbinom:
                param_n = mean * mean / (var - mean)
                param_p = mean / var
                params[i] = [param_n, param_p]
            else:
                params[i] = dist.fit(samples)
            rv_mixed = dist(*params[i])
            marginals[i] = Marginal(rv_mixed)
        # Calculate Akaike information criterion
        aic = np.zeros(len(options))
        for i, marginal in enumerate(marginals):
            aic[i] = 2 * len(params[i]) \
                     - 2 * np.sum(marginal.logpdf(samples))
        best_marginal = marginals[np.argmin(aic)]
        return best_marginal
