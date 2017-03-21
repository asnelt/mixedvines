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
from scipy.optimize import minimize
from scipy.stats import norm, gamma, poisson, binom, nbinom
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

    def logpdf(self, samples):
        '''
        Calculates the log of the probability density function.
        '''
        if self.is_continuous:
            return self.rv_mixed.logpdf(samples)
        else:
            return self.rv_mixed.logpmf(samples)

    def pdf(self, samples):
        '''
        Calculates the probability density function.
        '''
        return np.exp(self.logpdf(samples))

    def logcdf(self, samples):
        '''
        Calculates the log of the cumulative distribution function.
        '''
        return self.rv_mixed.logcdf(samples)

    def cdf(self, samples):
        '''
        Calculates the cumulative distribution function.
        '''
        return np.exp(self.logcdf(samples))

    def ppf(self, samples):
        '''
        Calculates the inverse of the cumulative distribution function.
        '''
        return self.rv_mixed.ppf(samples)

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
            marginals[i] = Marginal(rv_mixed, is_continuous)
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
        This class represents a layer of a copula vine tree.  A tree
        description in layers is advantageous, because most operations on the
        vine work in sweeps from layer to layer.
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
            # Set indices of input marginals
            if input_indices:
                if input_layer.is_marginal_layer():
                    self.input_marginal_indices = input_indices
                else:
                    self.input_marginal_indices = []
                    for _, i_ind in enumerate(input_indices):
                        self.input_marginal_indices.append(np.array([
                            input_layer.input_marginal_indices[i_ind[0]][1],
                            input_layer.input_marginal_indices[i_ind[1]][1]]))
            else:
                self.input_marginal_indices = None

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

        def logpdf(self, samples):
            '''
            Calculates the log of the probability density function.
            '''
            if self.is_root_layer():
                res = self.densities(samples)
                return res[0]
            else:
                return self.output_layer.logpdf(samples)

        def densities(self, samples):
            '''
            Computes densities and cumulative distribution functions layer by
            layer.
            '''
            if self.is_marginal_layer():
                # Evaluate marginal densities
                logp = np.zeros(samples.shape)
                cdfp = np.zeros(samples.shape)
                cdfm = np.zeros(samples.shape)
                is_continuous = np.zeros(len(self.marginals), dtype=bool)
                for k, marginal in enumerate(self.marginals):
                    is_continuous[k] = marginal.is_continuous
                    cdfp[:, k] = marginal.cdf(samples[:, k])
                    if marginal.is_continuous:
                        logp[:, k] = marginal.logpdf(samples[:, k])
                    else:
                        cdfm[:, k] = marginal.cdf(samples[:, k] - 1)
                        logp[:, k] = np.log(cdfp[:, k] - cdfm[:, k])
                logpdf = logp[:, 0]
            else:
                # Propagate samples to input_layer
                (logpdf_in, logp_in, cdfp_in, cdfm_in, is_continuous_in) \
                    = self.input_layer.densities(samples)
                logp = np.zeros((samples.shape[0], len(self.copulas)))
                cdfp = np.zeros((samples.shape[0], len(self.copulas)))
                cdfm = np.zeros((samples.shape[0], len(self.copulas)))
                is_continuous = np.zeros(len(self.copulas), dtype=bool)
                for k, copula in enumerate(self.copulas):
                    i = self.input_indices[k][0]
                    j = self.input_indices[k][1]
                    if is_continuous_in[i] and is_continuous_in[j]:
                        cdfp[:, k] = copula.ccdf(np.array([cdfp_in[:, i],
                                                           cdfp_in[:, j]]).T,
                                                 axis=0)
                        log_c = copula.logpdf(np.array([cdfp_in[:, i],
                                                        cdfp_in[:, j]]).T)
                        logp[:, k] = log_c + logp_in[:, j]
                    elif not is_continuous_in[i] and is_continuous_in[j]:
                        cdf0 = copula.cdf(np.array([cdfp_in[:, i],
                                                    cdfp_in[:, j]]).T)
                        cdf1 = copula.cdf(np.array([cdfm_in[:, i],
                                                    cdfp_in[:, j]]).T)
                        cdfp[:, k] = np.exp(np.log(cdf0 - cdf1)
                                            - logp_in[:, i])
                        pdf0 = copula.ccdf(np.array([cdfp_in[:, i],
                                                     cdfp_in[:, j]]).T)
                        pdf1 = copula.ccdf(np.array([cdfm_in[:, i],
                                                     cdfp_in[:, j]]).T)
                        logp[:, k] = np.log(pdf0 - pdf1) + logp_in[:, j] \
                            - logp_in[:, i]
                    elif is_continuous_in[i] and not is_continuous_in[j]:
                        cdfp[:, k] = copula.ccdf(np.array([cdfp_in[:, i],
                                                           cdfp_in[:, j]]).T,
                                                 axis=0)
                        cdfm[:, k] = copula.ccdf(np.array([cdfp_in[:, i],
                                                           cdfm_in[:, j]]).T,
                                                 axis=0)
                        logp[:, k] = np.log(cdfp[:, k] - cdfm[:, k])
                    else:
                        cdf0 = copula.cdf(np.array([cdfp_in[:, i],
                                                    cdfp_in[:, j]]).T)
                        cdf1 = copula.cdf(np.array([cdfm_in[:, i],
                                                    cdfp_in[:, j]]).T)
                        cdfp[:, k] = np.exp(np.log(cdf0 - cdf1)
                                            - logp_in[:, i])
                        cdf0 = copula.cdf(np.array([cdfp_in[:, i],
                                                    cdfm_in[:, j]]).T)
                        cdf1 = copula.cdf(np.array([cdfm_in[:, i],
                                                    cdfm_in[:, j]]).T)
                        cdfm[:, k] = np.exp(np.log(cdf0 - cdf1)
                                            - logp_in[:, i])
                        logp[:, k] = np.log(cdfp[:, k] - cdfm[:, k])
                    # This propagation of continuity is specific for the c-vine
                    is_continuous[k] = is_continuous_in[j]
                logpdf = logpdf_in + logp[:, 0]
            return (logpdf, logp, cdfp, cdfm, is_continuous)

        def build_curvs(self, urvs, curvs):
            '''
            Helper function for `make_dependent`.  Builds conditional uniform
            random variates `curvs` for `make_dependent`.
            '''
            (urvs, curvs) = self.make_dependent(urvs, curvs)
            if self.is_marginal_layer():
                curvs[:, 0] = urvs[:, 0]
            else:
                copula_index = 0
                curv_index = self.input_marginal_indices[copula_index][1]
                curvs[:, curv_index] = self.curv_ccdf(urvs[:, curv_index],
                                                      curvs, copula_index)
            return (urvs, curvs)

        def curv_ccdf(self, sample, curvs, copula_index):
            '''
            Helper function for `build_cond` to generate a conditional sample.
            '''
            if not self.is_marginal_layer():
                sample = self.input_layer.curv_ccdf(
                    sample, curvs, self.input_indices[copula_index][1])
                curv_index = self.input_marginal_indices[copula_index][0]
                input_urvs = np.array([curvs[:, curv_index], sample]).T
                sample = self.copulas[copula_index].ccdf(input_urvs, axis=0)
            return sample

        def make_dependent(self, urvs, curvs=None):
            '''
            Helper function for `rvs`.  Introduces dependencies between the
            uniform random variates `urvs` according to the vine copula tree.
            '''
            if not curvs:
                curvs = np.zeros(shape=urvs.shape)
            if not self.is_marginal_layer():
                (urvs, curvs) = self.input_layer.build_curvs(urvs, curvs)
            copula_index = 0
            layer = self
            while not layer.is_marginal_layer():
                imi = layer.input_marginal_indices[copula_index]
                input_urvs = np.array([curvs[:, imi[0]], urvs[:, imi[1]]]).T
                urvs[:, imi[1]] = layer.copulas[copula_index].ppcf(input_urvs,
                                                                   axis=0)
                copula_index = layer.input_indices[copula_index][1]
                layer = layer.input_layer
            return (urvs, curvs)

        def rvs(self, size=1):
            '''
            Generates random variates from the mixed vine.  Currently assumes a
            c-vine structure.
            '''
            if self.is_root_layer():
                # Determine distribution dimension
                layer = self
                while not layer.is_marginal_layer():
                    layer = layer.input_layer
                dim = len(layer.marginals)
                samples = np.random.rand(shape=[size, dim])
                (samples, _) = self.make_dependent(samples)
                # Use marginals to transform dependent uniform samples
                for i, marginal in enumerate(layer.marginals):
                    samples[:, i] = marginal.ppf(samples[:, i])
                return samples
            else:
                return self.output_layer.rvs(size)

        def fit(self, samples, is_continuous, trunc_level=None):
            '''
            Fits a vine tree.
            '''
            if self.is_marginal_layer():
                output_urvs = np.zeros(samples.shape)
                for i in range(samples.shape[1]):
                    self.marginals[i] = Marginal.fit(samples[:, i],
                                                     is_continuous[i])
                    output_urvs[:, i] = self.marginals[i].cdf(samples[:, i])
            else:
                input_urvs = self.input_layer.fit(samples, is_continuous)
                truncate = trunc_level and samples.shape[1] \
                    - len(self.input_indices) > trunc_level - 1
                output_urvs = np.zeros((samples.shape[0],
                                        len(self.input_indices)))
                self.copulas = []
                for i, i_ind in enumerate(self.input_indices):
                    if truncate:
                        next_copula = Copula('ind')
                    else:
                        next_copula = Copula.fit(input_urvs[:, i_ind])
                    self.copulas.append(next_copula)
                    output_urvs[:, i] = next_copula.ccdf(input_urvs[:, i_ind])
            return output_urvs

        def get_all_params(self):
            '''
            Constructs high dimensional vector containing all copula
            parameters.
            '''
            if self.is_marginal_layer():
                params = []
            else:
                params = self.input_layer.get_all_params()
                for copula in self.copulas:
                    for param in copula.theta:
                        params.append(param)
            return params

        def set_all_params(self, params):
            '''
            Sets all copula parameters to the values stored in params.
            '''
            if not self.is_marginal_layer():
                self.input_layer.set_all_params(params)
                if self.copulas:
                    for i in range(len(self.copulas)):
                        if self.copulas[i].theta:
                            for j in range(len(self.copulas[i].theta)):
                                self.copulas[i].theta[j] = params.pop(0)

        def get_all_bounds(self):
            '''
            Collects the bounds of all copula parameters.
            '''
            if self.is_marginal_layer():
                bnds = []
            else:
                bnds = self.input_layer.get_all_bounds()
                for copula in self.copulas:
                    for bnd in Copula.theta_bounds(copula.family):
                        bnds.append(bnd)
            return bnds

    def __init__(self, root=None, vine_type='c-vine'):
        '''
        Constructs a mixed vine model from a VineLayer root.
        '''
        self.root = root
        self.vine_type = vine_type

    def logpdf(self, samples):
        '''
        Calculates the log of the probability density function.
        '''
        return self.root.logpdf(samples)

    def pdf(self, samples):
        '''
        Calculates the probability density function.
        '''
        return np.exp(self.logpdf(samples))

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
        ent = 0.0
        var_sum = 0.0
        k = 0
        while sem >= sem_tol:
            # Generate samples
            samples = self.rvs(mc_size)
            logp = self.logpdf(samples)
            log2p = logp[np.isfinite(logp)] / np.log(2)
            k += 1
            # Monte-Carlo estimate of entropy
            ent += (-np.mean(log2p) - ent) / k
            # Estimate standard error
            var_sum += np.sum((-log2p - ent) ** 2)
            sem = conf * np.sqrt(var_sum / (k * mc_size * (k * mc_size - 1)))
        return ent, sem

    @staticmethod
    def fit(samples, is_continuous, vine_type='c-vine', trunc_level=None,
            do_refine=False):
        '''
        Fits the mixed vine to the given samples.
        '''
        if vine_type != 'c-vine':
            raise NotImplementedError
        dim = samples.shape[1]
        root = MixedVine._construct_c_vine(dim)
        root.fit(samples, is_continuous, trunc_level)
        vine = MixedVine(root, vine_type)
        if do_refine:
            # Refine copula parameters
            initial_point = root.get_all_params()
            bnds = root.get_all_bounds()

            def cost(params):
                '''
                Calculates the cost of a given set of copula parameters.
                '''
                return MixedVine._params_cost(params, samples, vine)

            result = minimize(cost, initial_point, method='TNC', bounds=bnds)
            vine.root.set_all_params(result.x)
        return vine

    @staticmethod
    def _construct_c_vine(dim):
        '''
        Constructs a c-vine tree without fitting it.
        '''
        marginals = np.empty(dim, dtype=object)
        layer = MixedVine.VineLayer(marginals=marginals)
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

    @staticmethod
    def _params_cost(params, samples, vine):
        '''
        Helper function for copula parameter optimization.
        '''
        vine.root.set_all_params(params)
        return -np.sum(vine.logpdf(samples))
