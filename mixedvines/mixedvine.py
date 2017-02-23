# Copyright (C) 2017 Arno Onken
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
This module implements a copula vine model with mixed marginals.
'''
from __future__ import division
from scipy.stats import norm, gamma, poisson, binom
import numpy as np
from mixedvines.copula import Copula


class Marginal(object):
    '''
    This class represents a marginal distribution which can be continuous or
    discrete.
    '''
    def __init__(self, rv_mixed, is_continuous):
        '''
        Constructs a marginal distribution.
        '''
        self.rv_mixed = rv_mixed
        self.is_continuous = is_continuous

    def logpdf(self, x):
        '''
        Calculates the log of the probability density function.
        '''
        if self.is_continuous:
            return self.rv_mixed.logpdf(x)
        else:
            return self.rv_mixed.logpmf(x)

    def pdf(self, x):
        '''
        Calculates the probability density function.
        '''
        return np.exp(self.logpdf(x))

    def logcdf(self, x):
        '''
        Calculates the log of the cumulative distribution function.
        '''
        return self.rv_mixed.logcdf(x)

    def cdf(self, x):
        '''
        Calculates the cumulative distribution function.
        '''
        return np.exp(self.logcdf(x))

    def ppf(self, q):
        '''
        Calculates the inverse of the cumulative distribution function.
        '''
        return self.rv_mixed.ppf(q)

    def rvs(self, size=1):
        '''
        Generates random variates from the distribution.
        '''
        return self.rv_mixed.rvs(size)

    @staticmethod
    def fit(samples, is_continuous):
        '''
        Fits a distribution to the given samples.
        '''
        # Fit parameters
        if is_continuous:
            # Suitable continuous distributions
            if np.any(samples <= 0):
                options = [norm]
            else:
                options = [norm, gamma]
            params = [dist.fit(samples) for dist in options]
        else:
            # Suitable discrete distributions
            options = [poisson, binom]
            params = np.empty(len(options))
            # Fit Poisson parameter
            params[options.index(poisson)] = [np.mean(samples)]
            # Fit binomial parameters
            binom_n = np.max(samples)
            binom_p = np.sum(samples) / (binom_n * len(samples))
            params[options.index(binom)] = [binom_n, binom_p]
        # Construct marginals
        marginals = []
        for i, dist in enumerate(options):
            rv_mixed = dist(*params[i])
            marginals.append(Marginal(rv_mixed, is_continuous))
        # Calculate Akaike information criterion
        aic = np.zeros(len(options))
        for i, marginal in enumerate(marginals):
            aic[i] = 2 * len(params[i]) \
                     - 2 * np.sum(marginal.logpdf(samples))
        best_marginal = marginals[np.argmin(aic)]
        return best_marginal


class MixedVine(object):
    '''
    This class represents a vine model with mixed marginals.
    '''
    def __init__(self, copulas, marginals, vine_type="c-vine"):
        '''
        Constructs a mixed vine model from a copula and marginals.
        '''
        self.copulas = copulas
        self.marginals = marginals
        self.vine_type = vine_type

    def logpdf(self, x):
        '''
        Calculates the log of the probability density function.
        '''
        raise NotImplementedError

    def pdf(self, x):
        '''
        Calculates the probability density function.
        '''
        return np.exp(self.logpdf(x))

    def rvs(self, size=1):
        '''
        Generates random variates from the mixed vine.
        '''
        d = len(self.marginals)
        w = np.random.rand(size, d)
        v = np.zeros(shape=[size, d, d])
        v[:, 0, 0] = w[:, 0]
        for i in range(1, d):
            v[:, i, 0] = np.reshape(w[:, i], [size, 1, 1])
            for k in reversed(range(i-1)):
                v[:, i, 1] = self.copulas[k, i].ppcf(
                        np.array([v[:, k, k], v[:, i, 0]]).T, axis=0)
            if i < d:
                for j in range(i-1):
                    v[:, i, j+1] = self.copulas[j, i].ccdf(
                            np.array([v[:, j, j], v[:, i, j]]).T, axis=0)
        u = v[:, :, 0]
        x = np.zeros(shape=u.shape)
        for i, marginal in enumerate(self.marginals):
            x[:, i] = marginal.ppf(u[:, i])
        return x

    def entropy(self, alpha=0.05, sem_tol=1e-3, mc_size=1000):
        '''
        Estimates the entropy of the mixed vine.
        '''
        # Gaussian confidence interval for sem_tol and level alpha
        conf = norm.ppf(1 - alpha)
        sem = np.inf
        h = 0.0
        var_sum = 0.0
        k = 0
        while sem >= sem_tol:
            # Generate samples
            x = self.rvs(mc_size)
            logp = self.logpdf(x)
            log2p = logp[np.isfinite(logp)] / np.log(2)
            k += 1
            # Monte-Carlo estimate of entropy
            h += (-np.mean(log2p) - h) / k
            # Estimate standard error
            var_sum += np.sum((-log2p - h) ** 2)
            sem = conf * np.sqrt(var_sum / (k * mc_size * (k * mc_size - 1)))
        return h, sem

    @staticmethod
    def fit(samples, is_continuous, vine_type="c-vine", trunc=None,
            do_refine=False):
        '''
        Fits the mixed vine to the given samples.
        '''
        raise NotImplementedError
        marginals = []
        for i in range(samples.shape[1]):
            marginals.append(Marginal.fit(samples[:, i], is_continuous[i]))
        copulas = []
        return MixedVine(copulas, marginals, vine_type)
