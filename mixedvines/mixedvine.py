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

    class VineLayer(object):
        '''
        This class represents a layer of a copula vine tree. A tree description
        in layers is advantageous, because most operations on the vine work in
        sweeps from layer to layer.
        '''
        def __init__(self, input_layer=None, input_indices=None,
                     marginals=None, copulas=None):
            '''
            Constructs a layer of a copula vine tree.
            '''
            self.input_layer = input_layer
            self.output_layer = None
            if input_layer:
                input_layer.output_layer = self
            self.input_indices = input_indices
            self.marginals = marginals
            self.copulas = copulas

        def is_marginal_layer(self):
            '''
            Determines whether the layer is the marginal layer.
            '''
            return not self.input_layer

        def is_root_layer(self):
            '''
            Determines whether the layer is the output layer.
            '''
            return not self.output_layer

        def _build_diagonal(self, last_index, u, v):
            '''
            '''
            self._build_samples(last_index, u, v)
            self._post_transform(u[:, last_index], v, last_index, 0)

        def _post_transform(self, t, v, i, k):
            '''
            '''
            if not self.is_marginal_layer():
                t = self.input_layer._post_transform(t, v, i, k + 1)
            u_sub = np.array([v[:, i], t]).T
            return self.copulas[k].ccdf(u_sub)

        def _build_samples(self, last_index, u, v):
            '''
            '''
            self.input_layer._build_diagonal(last_index - 1, u, v)
            u[:, last_index] = np.random.rand(u.shape[0])
            layer = self
            k = 0
            while not layer.is_marginal_layer():
                u_sub = np.array([v[:, last_index - k - 1],
                                  u[:, last_index]]).T
                u[:, last_index] = layer.copulas[k].ppcf(u_sub)
                layer = layer.input_layer
                k += 1

        def rvs(self, size):
            '''
            Currently ignores input_indices. Works only for c-vine.
            '''
            if self.is_root_layer():
                # Determine distribution dimension
                layer = self
                while not layer.is_marginal_layer():
                    layer = layer.input_layer
                dim = len(layer.marginals)
                last_index = dim - 1
                u = np.zeros(shape=[size, dim])
                v = np.zeros(shape=[size, dim])
                self._build_samples(last_index, u, v)
                # Use marginals to transform dependent uniform samples
                for i, marginal in enumerate(layer.marginals):
                    u[:, i] = marginal.ppf(u[:, i])
                return u
            else:
                return self.output_layer.rvs(size)

        def fit(self, samples, is_continuous, trunc_level=None):
            '''
            Fits a vine tree.
            '''
            if self.is_marginal_layer():
                self.marginals = []
                output_u = np.zeros(samples.shape)
                for i in range(samples.shape[1]):
                    self.marginals.append(Marginal.fit(samples[:, i],
                                                       is_continuous[i]))
                    output_u[:, i] = self.marginals[i].cdf(samples[:, i])
            else:
                input_u = self.input_layer.fit(samples, is_continuous)
                truncate = trunc_level and samples.shape[1] \
                    - len(self.input_indices) > trunc_level - 1
                output_u = np.zeros((samples.shape[0],
                                     len(self.input_indices)))
                self.copulas = []
                for i, i_ind in enumerate(self.input_indices):
                    if truncate:
                        next_copula = Copula('ind')
                    else:
                        next_copula = Copula.fit(input_u[:, i_ind])
                    self.copulas.append(next_copula)
                    output_u[:, i] = next_copula.ccdf(input_u[:, i_ind])
            return output_u

    def __init__(self, root=None, vine_type='c-vine'):
        '''
        Constructs a mixed vine model from a VineLayer root.
        '''
        self.root = root
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
        return self.root.rvs(size)

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
    def fit(samples, is_continuous, vine_type='c-vine', trunc_level=None,
            do_refine=False):
        '''
        Fits the mixed vine to the given samples.
        '''
        if vine_type != 'c-vine' or do_refine:
            raise NotImplementedError
        dim = samples.shape[1]
        root = MixedVine._construct_c_vine(dim)
        root.fit(samples, is_continuous, trunc_level)
        return MixedVine(root, vine_type)

    @staticmethod
    def _construct_c_vine(dim):
        '''
        Constructs a c-vine tree without fitting it.
        '''
        layer = MixedVine.VineLayer()
        for i in range(1, dim):
            input_indices = []
            # For each successor layer, generate c-vine input indices
            for j in range(dim - i):
                input_indices.append(np.array([0, j+1]))
            # Generate vine layer
            layer = MixedVine.VineLayer(input_layer=layer,
                                        input_indices=input_indices)
        root = layer
        return root
