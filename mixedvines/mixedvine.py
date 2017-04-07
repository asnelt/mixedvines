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
from mixedvines.copula import Copula, IndependenceCopula


class Marginal(object):
    '''
    This class represents a marginal distribution which can be continuous or
    discrete.

    Methods
    -------
    ``logpdf(samples)``
        Log of the probability density function or probability mass function.
    ``pdf(samples)``
        Probability density function or probability mass function.
    ``logcdf(samples)``
        Log of the cumulative distribution function.
    ``cdf(samples)``
        Cumulative distribution function.
    ``ppf(samples)``
        Inverse of the cumulative distribution function.
    ``rvs(size=1)``
        Generate random variates.
    ``fit(samples)``
        Fit a distribution to samples.
    '''

    def __init__(self, rv_mixed, is_continuous):
        '''
        Constructs a marginal distribution.

        Parameters
        ----------
        rv_mixed : scipy.stats.rv_continuous or scipy.stats.rv_discrete
            The distribution object, either of a continuous or of a discrete
            univariate distribution.
        is_continuous : bool
            If `true` then `rv_mixed` is a continuous distribution.  Otherwise,
            `rv_mixed` is a discrete distribution.
        '''
        self.rv_mixed = rv_mixed
        self.is_continuous = is_continuous

    def logpdf(self, samples):
        '''
        Calculates the log of the probability density function.

        Parameters
        ----------
        samples : array_like
            Array of samples.

        Returns
        -------
        vals : ndarray
            Log of the probability density function evaluated at `samples`.
        '''
        if self.is_continuous:
            return self.rv_mixed.logpdf(samples)
        else:
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
        vals : ndarray
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
        vals : ndarray
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
        vals : ndarray
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
        vals : ndarray
            Inverse of the cumulative distribution function evaluated at
            `samples`.
        '''
        return self.rv_mixed.ppf(samples)

    def rvs(self, size=1):
        '''
        Generates random variates from the distribution.

        Parameters
        ----------
        size : integer, optional
            The number of samples to generate.  (Default: 1)

        Returns
        -------
        samples : array_like
            Array of samples.
        '''
        return self.rv_mixed.rvs(size)

    @staticmethod
    def fit(samples, is_continuous):
        '''
        Fits a distribution to the given samples.

        Parameters
        ----------
        samples : array_like
            Array of samples.
        is_continuous : bool
            If `true` then a continuous distribution is fitted.  Otherwise, a
            discrete distribution is fitted.

        Returns
        -------
        marginal : Marginal
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
    This class represents a copula vine model with mixed marginals.

    Methods
    -------
    ``logpdf(samples)``
        Calculates the log of the probability density function.
    ``pdf(samples)``
        Calculates the probability density function.
    ``rvs(size)``
        Generates random variates from the mixed vine.
    ``entropy(alpha, sem_tol, mc_size)``
        Estimates the entropy of the mixed vine.
    ``fit(samples, is_continuous, vine_type, trunc_level, do_refine)``
        Fits the mixed vine to the given samples.
    '''

    class VineLayer(object):
        '''
        This class represents a layer of a copula vine tree.  A tree
        description in layers is advantageous, because most operations on the
        vine work in sweeps from layer to layer.

        Methods
        -------
        ``is_marginal_layer()``
            Determines whether the layer is a marginal layer.
        ``is_root_layer()``
            Determines whether the layer is a root layer.
        ``logpdf(samples)``
            Log of the probability density function.
        ``densities(samples)``
            Computes densities and cumulative distribution functions.
        ``build_curvs(urvs, curvs)``
            Builds conditional uniform random variates `curvs` for
            `make_dependent`.
        ``curv_ccdf(sample, curvs, copula_index)``
            Generates a conditional sample for `build_curvs`.
        ``make_dependent(urvs, curvs)``
            Introduces dependencies between the uniform random variates `urvs`.
        ``rvs(size)``
            Generates random variates from the mixed vine.
        ``fit(samples, is_continuous, trunc_level)``
            Fits a vine tree.
        ``get_all_params()``
            Constructs an array containing all copula parameters.
        ``set_all_params(params)``
            Sets all copula parameters to the values stored in params.
        ``get_all_bounds()``
            Collects the bounds of all copula parameters.
        '''

        def __init__(self, input_layer=None, input_indices=None,
                     marginals=None, copulas=None):
            '''
            Constructs a layer of a copula vine tree.

            Parameters
            ----------
            input_layer : VineLayer, optional
                The layer providing input.  (Default: None)
            input_indices : array_like, optional
                Array of length n where n is the number of copulas in this
                layer.  Each element in the array is a 2-tuple containing the
                left and right input indices of the respective pair-copula.
                `None` if this is the marginal layer.
            marginals : array_like, optional
                List with the marginal distributions as elements.  `None` if
                this is not the marginal layer.
            copulas : array_like, optional
                List with the pair-copulas of this layer as elements.  `None`
                if this is the marginal layer.
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

            Returns
            -------
            iml : boolean
                `True` if the layer is the marginal layer.
            '''
            return not self.input_layer

        def is_root_layer(self):
            '''
            Determines whether the layer is the output layer.

            Returns
            -------
            irl : boolean
                `True` if the layer is the root layer.
            '''
            return not self.output_layer

        def logpdf(self, samples):
            '''
            Calculates the log of the probability density function.

            Parameters
            ----------
            samples : array_like
                n-by-d matrix of samples where n is the number of samples and d
                is the number of marginals.

            Returns
            -------
            vals : ndarray
                Log of the probability density function evaluated at `samples`.
            '''
            if self.is_root_layer():
                res = self.densities(samples)
                return res['logpdf']
            else:
                return self.output_layer.logpdf(samples)

        def _marginal_densities(self, samples):
            '''
            Evaluate marginal densities and cumulative distribution functions.

            Parameters
            ----------
            samples : array_like
                n-by-d matrix of samples where n is the number of samples and d
                is the number of marginals.

            Returns
            -------
            dout : dictionary
                The densities and cumulative distribution functions.  Keys:
                `logpdf`: Equal to first element of `logp`.
                'logp': Log of the probability density function.
                'cdfp': Upper cumulative distribution functions.
                'cdfm': Lower cumulative distribution functions.
                'is_continuous': List of booleans where element i is `True` if
                output element i is continuous.
            '''
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
            dout = {'logpdf': logpdf, 'logp': logp, 'cdfp': cdfp, 'cdfm': cdfm,
                    'is_continuous': is_continuous}
            return dout

        def densities(self, samples):
            '''
            Computes densities and cumulative distribution functions layer by
            layer.

            Parameters
            ----------
            samples : array_like
                n-by-d matrix of samples where n is the number of samples and d
                is the number of marginals.

            Returns
            -------
            dout : dictionary
                The densities and cumulative distribution functions.  Keys:
                `logpdf`: Sum of the first elements of `logp` of all input
                layers and this one.
                'logp': Log of the probability density function.
                'cdfp': Upper cumulative distribution functions.
                'cdfm': Lower cumulative distribution functions.
                'is_continuous': List of booleans where element i is `True` if
                output element i is continuous.
            '''
            if self.is_marginal_layer():
                return self._marginal_densities(samples)
            else:
                # Propagate samples to input_layer
                din = self.input_layer.densities(samples)
                # Prepare output densities
                logp = np.zeros((samples.shape[0], len(self.copulas)))
                cdfp = np.zeros((samples.shape[0], len(self.copulas)))
                cdfm = np.zeros((samples.shape[0], len(self.copulas)))
                is_continuous = np.zeros(len(self.copulas), dtype=bool)
                for k, copula in enumerate(self.copulas):
                    i = self.input_indices[k][0]
                    j = self.input_indices[k][1]
                    # Distinguish between discrete and continuous inputs
                    if din['is_continuous'][i] and din['is_continuous'][j]:
                        cdfp[:, k] \
                            = copula.ccdf(
                                np.array([din['cdfp'][:, i],
                                          din['cdfp'][:, j]]).T, axis=0)
                        logp[:, k] \
                            = copula.logpdf(
                                np.array([din['cdfp'][:, i],
                                          din['cdfp'][:, j]]).T) \
                            + din['logp'][:, j]
                    elif not din['is_continuous'][i] \
                            and din['is_continuous'][j]:
                        cdfp[:, k] \
                            = np.exp(np.log(
                                copula.cdf(
                                    np.array([din['cdfp'][:, i],
                                              din['cdfp'][:, j]]).T)
                                - copula.cdf(
                                    np.array([din['cdfm'][:, i],
                                              din['cdfp'][:, j]]).T))
                                     - din['logp'][:, i])
                        logp[:, k] \
                            = np.log(
                                copula.ccdf(
                                    np.array([din['cdfp'][:, i],
                                              din['cdfp'][:, j]]).T)
                                - copula.ccdf(
                                    np.array([din['cdfm'][:, i],
                                              din['cdfp'][:, j]]).T)) \
                            - din['logp'][:, i] + din['logp'][:, j]
                    elif din['is_continuous'][i] \
                            and not din['is_continuous'][j]:
                        cdfp[:, k] \
                            = copula.ccdf(
                                np.array([din['cdfp'][:, i],
                                          din['cdfp'][:, j]]).T, axis=0)
                        cdfm[:, k] \
                            = copula.ccdf(
                                np.array([din['cdfp'][:, i],
                                          din['cdfm'][:, j]]).T, axis=0)
                        logp[:, k] = np.log(cdfp[:, k] - cdfm[:, k])
                    else:
                        cdfp[:, k] \
                            = np.exp(np.log(
                                copula.cdf(
                                    np.array([din['cdfp'][:, i],
                                              din['cdfp'][:, j]]).T)
                                - copula.cdf(
                                    np.array([din['cdfm'][:, i],
                                              din['cdfp'][:, j]]).T))
                                     - din['logp'][:, i])
                        cdfm[:, k] \
                            = np.exp(np.log(
                                copula.cdf(
                                    np.array([din['cdfp'][:, i],
                                              din['cdfm'][:, j]]).T)
                                - copula.cdf(
                                    np.array([din['cdfm'][:, i],
                                              din['cdfm'][:, j]]).T))
                                     - din['logp'][:, i])
                        logp[:, k] = np.log(cdfp[:, k] - cdfm[:, k])
                    # This propagation of continuity is specific for the c-vine
                    is_continuous[k] = din['is_continuous'][j]
                logpdf = din['logpdf'] + logp[:, 0]
            dout = {'logpdf': logpdf, 'logp': logp, 'cdfp': cdfp, 'cdfm': cdfm,
                    'is_continuous': is_continuous}
            return dout

        def build_curvs(self, urvs, curvs):
            '''
            Helper function for `make_dependent`.  Builds conditional uniform
            random variates `curvs` for `make_dependent`.

            Parameters
            ----------
            urvs : array_like
                Uniform random variates to be made dependent by
                `make_dependent`.
            curvs : array_like
                Array to be filled with dependent conditional uniform random
                variates by `make_dependent`.

            Returns
            -------
            urvs : array_like
                Dependent uniform random variates.
            curvs : array_like
                Conditional uniform random variates.
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
            Helper function for `build_curvs` to generate a conditional sample.

            Parameters
            ----------
            sample : float
                Right input for the marginal layer.
            curvs : array_like
                Conditional uniform random variates.
            copula_index : integer
                Index of the copula to be used to generate the dependent
                sample.

            Returns
            -------
            sample : float
                Conditional sample for `curvs` at index `copula_index`.
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

            Parameters
            ----------
            urvs : array_like
                Uniform random variates to be made dependent.
            curvs : array_like, optional
                Array to be filled with dependent conditional uniform random
                variates by `build_curvs'.  (Default: None)

            Returns
            -------
            urvs : array_like
                Dependent uniform random variates.
            curvs : array_like
                Conditional uniform random variates.
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

            Parameters
            ----------
            size : integer, optional
                The number of samples to generate.  (Default: 1)

            Returns
            -------
            samples : array_like
                n-by-d matrix of samples where n is the number of samples and d
                is the number of marginals.
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
            Fits the vine tree to the given samples.  This method is supposed
            to be called on the output layer and will recurse to its input
            layers.

            Parameters
            ----------
            samples : array_like
                n-by-d matrix of samples where n is the number of samples and d
                is the number of marginals.
            is_continuous : array_like
                List of boolean values, where element i is `True` if marginal i
                is continuous.
            trunc_level : integer, optional
                Layer level to truncate the vine at.  Copulas in layers beyond
                are just independence copulas.  If the level is `None`, then
                the vine is not truncated.  (Default: None)

            Returns
            -------
            output_urvs : array_like
                The output uniform random variates of the layer.  Can be
                ignored if this is the output layer.
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
                for i, i_ind in enumerate(self.input_indices):
                    if truncate:
                        self.copulas[i] = IndependenceCopula()
                    else:
                        self.copulas[i] = Copula.fit(input_urvs[:, i_ind])
                    output_urvs[:, i] = next_copula.ccdf(input_urvs[:, i_ind])
            return output_urvs

        def get_all_params(self):
            '''
            Constructs an array containing all copula parameters.

            Returns
            -------
            params : array_like
                A list containing all copula parameter values starting with the
                parameters of the first copula layer and continuing layer by
                layer.
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

            Parameters
            ----------
            params : array_like
                A list containing all copula parameter values starting with the
                parameters of the first copula layer and continuing layer by
                layer.
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

            Returns
            -------
            bnds : array_like
                A list of 2-tuples containing all copula parameter bounds
                starting with the parameters of the first copula layer and
                continuing layer by layer.  The first element of tuple i
                denotes the lower bound and the second element denotes the
                upper bound of parameter i.
            '''
            if self.is_marginal_layer():
                bnds = []
            else:
                bnds = self.input_layer.get_all_bounds()
                for copula in self.copulas:
                    for bnd in copula.theta_bounds():
                        bnds.append(bnd)
            return bnds

    def __init__(self, dim, vine_type='c-vine'):
        '''
        Constructs a mixed vine model.

        Parameters
        ----------
        dim : integer
            The number of marginals of the vine model.  Must be greater than 1.
        vine_type : string, optional
            Type of the vine tree.  Currently, only the canonical vine
            ('c_vine') is supported.  (Default: 'c-vine')
        '''
        if dim < 2:
            raise ValueError("The number of marginals 'dim' must be greater"
                             " than 1.")
        if vine_type != 'c-vine':
            raise NotImplementedError
        self.vine_type = vine_type
        self.root = self._construct_c_vine(dim)

    def logpdf(self, samples):
        '''
        Calculates the log of the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d is
            the number of marginals.

        Returns
        -------
        vals : ndarray
            Log of the probability density function evaluated at `samples`.
        '''
        return self.root.logpdf(samples)

    def pdf(self, samples):
        '''
        Calculates the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.

        Returns
        -------
        vals : ndarray
            Probability density function evaluated at `samples`.
        '''
        return np.exp(self.logpdf(samples))

    def rvs(self, size=1):
        '''
        Generates random variates from the mixed vine.

        Parameters
        ----------
        size : integer, optional
            The number of samples to generate.  (Default: 1)

        Returns
        -------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d is
            the number of marginals.
        '''
        return self.root.rvs(size)

    def entropy(self, alpha=0.05, sem_tol=1e-3, mc_size=1000):
        '''
        Estimates the entropy of the mixed vine.

        Parameters
        ----------
        alpha : float, optional
            Significance level of the entropy estimate.  (Default: 0.05)
        sem_tol : float, optional
            Maximum standard error as a stopping criterion.  (Default: 1e-3)
        mc_size : integer, optional
            Number of samples that are drawn in each iteration of the Monte
            Carlo estimation.  (Default: 1000)

        Returns
        -------
        ent : float
            Estimate of the mixed vine entropy in bits.
        sem : float
            Standard error of the mixed vine entropy estimate in bits.
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

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d is
            the number of marginals.
        is_continuous : array_like
            List of boolean values, where element i is `True` if marginal i is
            continuous.
        vine_type : string, optional
            Type of the vine tree.  Currently, only the canonical vine
            ('c_vine') is supported.  (Default: 'c-vine')
        trunc_level : integer, optional
            Layer level to truncate the vine at.  Copulas in layers beyond are
            just independence copulas.  If the level is `None`, then the vine
            is not truncated.  (Default: None)
        do_refine : boolean, optional
            If true, then all pair copula parameters are optimized jointly at
            the end.  (Default: False)

        Returns
        -------
        vine : MixedVine
            The mixed vine with parameters fitted to `samples`.
        '''
        if vine_type != 'c-vine':
            raise NotImplementedError
        dim = samples.shape[1]
        vine = MixedVine(dim, vine_type)
        vine.root.fit(samples, is_continuous, trunc_level)
        if do_refine:
            # Refine copula parameters
            initial_point = vine.root.get_all_params()
            bnds = vine.root.get_all_bounds()

            def cost(params):
                '''
                Calculates the cost of a given set of copula parameters.
                '''
                vine.root.set_all_params(params)
                vals = vine.logpdf(samples)
                return -np.sum(vals)

            result = minimize(cost, initial_point, method='TNC', bounds=bnds)
            vine.root.set_all_params(result.x)
        return vine

    @staticmethod
    def _construct_c_vine(dim):
        '''
        Constructs a c-vine tree without setting marginals or copulas.

        Parameters
        ----------
        dim : integer
            The number of marginals of the canonical vine tree.

        Returns
        -------
        root : VineLayer
            The root layer of the canonical vine tree.
        '''
        marginals = np.empty(dim, dtype=Marginal)
        layer = MixedVine.VineLayer(marginals=marginals)
        for i in range(1, dim):
            input_indices = []
            # For each successor layer, generate c-vine input indices
            for j in range(dim - i):
                input_indices.append(np.array([0, j+1]))
            copulas = np.empty(len(input_indices), dtype=Copula)
            # Generate vine layer
            layer = MixedVine.VineLayer(input_layer=layer,
                                        input_indices=input_indices,
                                        copulas=copulas)
        root = layer
        return root
