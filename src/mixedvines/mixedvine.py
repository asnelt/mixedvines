# Copyright (C) 2017-2019, 2021-2023 Arno Onken
#
# This file is part of the mixedvines package.
#
# The mixedvines package is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# The mixedvines package is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""This module implements a copula vine model with mixed marginals.

Classes
-------
MixedVine
    Copula vine model with mixed marginals.

"""
from scipy.stats import kendalltau, norm, uniform
from scipy.optimize import minimize
import numpy as np
from .marginal import Marginal
from .copula import Copula, IndependenceCopula


class MixedVine:
    """Represents a copula vine model with mixed marginals.

    Parameters
    ----------
    dim : int
        The number of marginals of the vine model.  Must be greater than 1.

    Raises
    ------
    ValueError
        If the number of marginals `dim` is not greater than 1.
    """

    def __init__(self, dim):
        if dim < 2:
            raise ValueError("the number of marginals 'dim' must be greater"
                             " than 1")
        self._root = self._construct_c_vine(np.arange(dim))

    def logpdf(self, samples):
        """Calculates the log of the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.

        Returns
        -------
        ndarray
            Log of the probability density function evaluated at `samples`.
        """
        return self._root.logpdf(samples)

    def pdf(self, samples):
        """Calculates the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.

        Returns
        -------
        ndarray
            Probability density function evaluated at `samples`.
        """
        return np.exp(self.logpdf(samples))

    def rvs(self, size=1, random_state=None):
        """Generates random variates from the mixed vine.

        Parameters
        ----------
        size : int, optional
            The number of samples to generate.  (Default: 1)
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            The random state to use for random variate generation.  `None`
            corresponds to the `RandomState` singleton.  For an `int`, a
            new `RandomState` is generated and seeded.  For a `RandomState`
            or `Generator`, the object is used.  (Default: `None`)

        Returns
        -------
        array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.
        """
        return self._root.rvs(size=size, random_state=random_state)

    def entropy(self, alpha=0.05, sem_tol=1e-3, mc_size=1000,
                random_state=None):
        """Estimates the entropy of the mixed vine.

        Parameters
        ----------
        alpha : float, optional
            Significance level of the entropy estimate.  (Default: 0.05)
        sem_tol : float, optional
            Maximum standard error as a stopping criterion.
            (Default: 1e-3)
        mc_size : int, optional
            Number of samples that are drawn in each iteration of the Monte
            Carlo estimation.  (Default: 1000)
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            The random state to use for random variate generation.  `None`
            corresponds to the `RandomState` singleton.  For an `int`, a
            new `RandomState` is genered and seeded.  For a `RandomState`
            or `Generator`, the object is used.  (Default: `None`)

        Returns
        -------
        ent : float
            Estimate of the mixed vine entropy in bits.
        sem : float
            Standard error of the mixed vine entropy estimate in bits.
        """
        # Gaussian confidence interval for sem_tol and level alpha
        conf = norm.ppf(1 - alpha)
        sem = np.inf
        ent = 0.0
        var_sum = 0.0
        k = 0
        while sem >= sem_tol:
            # Generate samples
            samples = self.rvs(size=mc_size, random_state=random_state)
            logp = self.logpdf(samples)
            log2p = logp[np.isfinite(logp)] / np.log(2)
            k += 1
            # Monte-Carlo estimate of entropy
            ent += (-np.mean(log2p) - ent) / k
            # Estimate standard error
            var_sum += np.sum((-log2p - ent) ** 2)
            sem = conf * np.sqrt(var_sum / (k * mc_size * (k * mc_size - 1)))
        return ent, sem

    def set_marginal(self, marginal_index, rv_mixed):
        """Sets a marginal distribution.

        Sets a particular marginal distribution in the mixed vine tree for
        manual construction of a mixed vine model.

        Parameters
        ----------
        marginal_index : int
            The index of the marginal in the marginal layer.
        rv_mixed : `scipy.stats.distributions.rv_frozen`
            The marginal distribution to be inserted.
        """
        marginal_layer = self._get_marginal_layer()
        marginal_layer.marginals[marginal_index] = Marginal(rv_mixed)

    def set_copula(self, layer_index, copula_index, copula):
        """Sets a pair-copula.

        Sets a particular pair-copula in the mixed vine tree for manual
        construction of a mixed vine model.

        Parameters
        ----------
        layer_index : int
            The index of the vine layer.
        copula_index : int
            The index of the copula in its layer.
        copula : Copula
            The copula to be inserted.

        Raises
        ------
        IndexError
            If the argument `layer_index` is out of range.
        """
        layer = self._get_marginal_layer()
        if layer_index < 1 or layer_index >= len(layer.marginals):
            raise IndexError("argument 'layer_index' out of range")
        for _ in range(layer_index):
            layer = layer.output_layer
        layer.copulas[copula_index] = copula

    def is_continuous(self):
        """Determines which marginals are continuous.

        Returns
        -------
        array_like
            List of boolean values of length d, where d is the number of
            marginals and element i is `True` if marginal i is continuous.
        """
        return self._root.is_continuous()

    @staticmethod
    def fit(samples, is_continuous, trunc_level=None, do_refine=False,
            keep_order=False):
        """Fits the mixed vine to the given samples.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.
        is_continuous : array_like
            List of boolean values of length d, where d is the number of
            marginals and element i is `True` if marginal i is continuous.
        trunc_level : int, optional
            Layer level to truncate the vine at.  Copulas in layers beyond
            are just independence copulas.  If the level is `None`, then
            the vine is not truncated.  (Default: `None`)
        do_refine : boolean, optional
            If `True`, then all pair-copula parameters are optimized
            jointly at the end.  (Default: `False`)
        keep_order : boolean, optional
            If `False`, then a heuristic is used to select the vine
            structure.  (Default: `False`)

        Returns
        -------
        vine : MixedVine
            The mixed vine with parameters fitted to `samples`.
        """
        dim = samples.shape[1]
        vine = MixedVine(dim)
        if not keep_order:
            element_order = MixedVine._heuristic_element_order(samples)
            vine._root = MixedVine._construct_c_vine(element_order)
        vine._root.fit(samples, is_continuous, trunc_level)
        if do_refine:
            # Refine copula parameters
            initial_point = vine._root.get_all_params()
            bnds = vine._root.get_all_bounds()

            def cost(params):
                """Calculates the cost of a set of copula parameters."""
                vine._root.set_all_params(params.tolist())
                vals = vine.logpdf(samples)
                return -np.sum(vals)

            result = minimize(cost, initial_point, method='SLSQP',
                              bounds=bnds)
            vine._root.set_all_params(result.x.tolist())
        return vine

    def _get_marginal_layer(self):
        """Returns the marginal layer of the MixedVine."""
        layer = self._root
        while not layer.is_marginal_layer():
            layer = layer.input_layer
        return layer

    @staticmethod
    def _heuristic_element_order(samples):
        """Finds a heuristic element order.

        Finds an order of elements that heuristically facilitates vine
        modeling.  For this purpose, Kendall's tau is calculated between
        samples of pairs of elements and elements are scored according to
        the sum of absolute Kendall's taus of pairs the elements appear in.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.

        Returns
        -------
        order : array_like
            Permutation of all element indices reflecting descending
            scores.
        """
        dim = samples.shape[1]
        # Score elements according to total absolute Kendall's tau
        score = np.zeros(dim)
        for i in range(1, dim):
            for j in range(i):
                tau = kendalltau(samples[:, i], samples[:, j]).correlation
                score[i] += np.abs(tau)
                score[j] += np.abs(tau)
        # Get order indices for descending score
        order = score.argsort()[::-1]
        return order

    @staticmethod
    def _construct_c_vine(element_order):
        """Constructs a c-vine.

        Constructs a c-vine tree without setting marginals or copulas.  The
        c-vine tree is constructed according to the input element order.
        The index of the element with the most important dependencies
        should come first in the input argument.

        Parameters
        ----------
        element_order : array_like
            Permutation of all element indices.

        Returns
        -------
        root : _VineLayer
            The root layer of the canonical vine tree.
        """
        dim = len(element_order)
        marginals = np.empty(dim, dtype=Marginal)
        layer = _VineLayer(marginals=marginals)
        identity_order = np.arange(dim - 1)
        for i in range(1, dim):
            if i == 1:
                order = element_order
            else:
                order = identity_order
            # For each successor layer, generate c-vine input indices
            input_indices = [np.array([order[0], order[j+1]])
                             for j in range(dim - i)]
            copulas = np.empty(len(input_indices), dtype=Copula)
            # Generate vine layer
            layer = _VineLayer(input_layer=layer,
                               input_indices=input_indices,
                               copulas=copulas)
        root = layer
        return root


class _VineLayer:
    """Represents a layer of a copula vine tree.

    A tree description in layers is advantageous, because most operations
    on the vine work in sweeps from layer to layer.

    Parameters
    ----------
    input_layer : _VineLayer, optional
        The layer providing input.  (Default: `None`)
    input_indices : array_like, optional
        Array of length n where n is the number of copulas in this layer.
        Each element in the array is a 2-tuple containing the left and
        right input indices of the respective pair-copula.  `None` if this
        is the marginal layer.
    marginals : array_like, optional
        List with the marginal distributions as elements.  `None` if this
        is not the marginal layer.
    copulas : array_like, optional
        List with the pair-copulas of this layer as elements.  `None` if
        this is the marginal layer.

    Attributes
    ----------
    input_layer : _VineLayer
        The layer providing input.
    output_layer : _VineLayer
        The output layer.
    input_indices : array_like
        Array with the input indices of the copulas.
    input_marginal_indices : array_like
        Array with the input indices of the marginals.
    marginals : array_like
        List with the marginal distributions of this layer as elements.
    copulas : array_like
        List with the pair-copulas of this layer as elements.
    """

    def __init__(self, input_layer=None, input_indices=None,
                 marginals=None, copulas=None):
        self.input_layer = input_layer
        self.output_layer = None
        if input_layer is not None:
            input_layer.output_layer = self
        self.input_indices = input_indices
        self.marginals = marginals
        self.copulas = copulas
        # Set indices of input marginals
        if input_indices is not None:
            if input_layer.is_marginal_layer():
                self.input_marginal_indices = input_indices
            else:
                self.input_marginal_indices = [np.array(
                    [input_layer.input_marginal_indices[i_ind[0]][1],
                     input_layer.input_marginal_indices[i_ind[1]][1]])
                    for i_ind in input_indices]
        else:
            self.input_marginal_indices = None

    def is_marginal_layer(self):
        """Determines whether the layer is the marginal layer.

        Returns
        -------
        boolean
            `True` if the layer is the marginal layer.
        """
        return self.input_layer is None

    def is_root_layer(self):
        """Determines whether the layer is the output layer.

        Returns
        -------
        boolean
            `True` if the layer is the root layer.
        """
        return self.output_layer is None

    def logpdf(self, samples):
        """Calculates the log of the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.

        Returns
        -------
        ndarray
            Log of the probability density function evaluated at
            `samples`.
        """
        if samples.size == 0:
            return np.empty((0, 1))
        if self.is_root_layer():
            res = self.densities(samples)
            return res['logpdf']
        return self.output_layer.logpdf(samples)

    def _marginal_densities(self, samples):
        """Evaluates marginal densities and CDFs.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.

        Returns
        -------
        dictionary
            The densities and cumulative distribution functions.  Keys:
            `logpdf`: Equal to first element of `logp`.
            'logp': Log of the probability density function.
            'cdfp': Upper cumulative distribution functions.
            'cdfm': Lower cumulative distribution functions.
            'is_continuous': List of booleans where element i is `True` if
                             output element i is continuous.
        """
        logp = np.zeros(samples.shape)
        cdfp = np.zeros_like(logp)
        cdfm = np.zeros_like(logp)
        for k, marginal in enumerate(self.marginals):
            cdfp[:, k] = marginal.cdf(samples[:, k])
            if marginal.is_continuous:
                logp[:, k] = marginal.logpdf(samples[:, k])
            else:
                cdfm[:, k] = marginal.cdf(samples[:, k] - 1)
                with np.errstate(divide='ignore'):
                    logp[:, k] = np.log(np.maximum(0, cdfp[:, k] - cdfm[:, k]))
        return {'logpdf': logp[:, self.output_layer.input_indices[0][0]],
                'logp': logp, 'cdfp': cdfp, 'cdfm': cdfm,
                'is_continuous': self.is_continuous()}

    def densities(self, samples):
        """Computes densities and cumulative distribution functions.

        The computation is done layer by layer.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.

        Returns
        -------
        dictionary
            The densities and cumulative distribution functions.  Keys:
            `logpdf`: Sum of the first elements of `logp` of all input
                      layers and this one.
            'logp': Log of the probability density function.
            'cdfp': Upper cumulative distribution functions.
            'cdfm': Lower cumulative distribution functions.
            'is_continuous': List of booleans where element i is `True` if
                             output element i is continuous.
        """
        if self.is_marginal_layer():
            return self._marginal_densities(samples)
        # Propagate samples to input_layer
        din = self.input_layer.densities(samples)
        # Prepare output density variables
        logp = np.zeros((samples.shape[0], len(self.copulas)))
        cdfp = np.zeros_like(logp)
        cdfm = np.zeros_like(logp)
        for k, copula in enumerate(self.copulas):
            i = self.input_indices[k][0]
            j = self.input_indices[k][1]
            cdfpp = np.array([din['cdfp'][:, i], din['cdfp'][:, j]]).T
            cdfpm = np.array([din['cdfp'][:, i], din['cdfm'][:, j]]).T
            cdfmp = np.array([din['cdfm'][:, i], din['cdfp'][:, j]]).T
            cdfmm = np.array([din['cdfm'][:, i], din['cdfm'][:, j]]).T
            isf = np.isfinite(din['logp'][:, i])
            # Distinguish between discrete and continuous inputs
            if din['is_continuous'][i] and din['is_continuous'][j]:
                cdfp[:, k] = copula.ccdf(cdfpp, axis=0)
                logp[:, k] = copula.logpdf(cdfpp) + din['logp'][:, j]
            elif not din['is_continuous'][i] and din['is_continuous'][j]:
                cdfp[~isf, k] = 0.0
                logp[~isf, k] = -np.inf
                with np.errstate(divide='ignore'):
                    cdfp[isf, k] = np.exp(np.log(
                        np.maximum(0, copula.cdf(cdfpp[isf, :])
                                   - copula.cdf(cdfmp[isf, :])))
                                          - din['logp'][isf, i])
                    logp[isf, k] = np.log(
                        np.maximum(0, copula.ccdf(cdfpp[isf, :])
                                   - copula.ccdf(cdfmp[isf, :]))
                        ) - din['logp'][isf, i] + din['logp'][isf, j]
            elif din['is_continuous'][i] and not din['is_continuous'][j]:
                cdfp[:, k] = copula.ccdf(cdfpp, axis=0)
                cdfm[:, k] = copula.ccdf(cdfpm, axis=0)
                with np.errstate(divide='ignore'):
                    logp[:, k] = np.log(np.maximum(0, cdfp[:, k] - cdfm[:, k]))
            else:
                cdfp[~isf, k] = 0.0
                cdfm[~isf, k] = 0.0
                with np.errstate(divide='ignore'):
                    cdfp[isf, k] = np.exp(np.log(
                        np.maximum(0, copula.cdf(cdfpp[isf, :])
                                   - copula.cdf(cdfmp[isf, :])))
                                          - din['logp'][isf, i])
                    cdfm[isf, k] = np.exp(np.log(
                        np.maximum(0, copula.cdf(cdfpm[isf, :])
                                   - copula.cdf(cdfmm[isf, :])))
                                          - din['logp'][isf, i])
                    logp[:, k] = np.log(np.maximum(0, cdfp[:, k] - cdfm[:, k]))
        return {'logpdf': din['logpdf'] + logp[:, 0],
                'logp': logp, 'cdfp': cdfp, 'cdfm': cdfm,
                # This propagation of continuity is specific to the c-vine
                'is_continuous':
                    [din['is_continuous'][second_input]
                     for second_input in list(zip(*self.input_indices))[1]]}

    def build_curvs(self, urvs, curvs):
        """Helper function for `_make_dependent`.

        Builds conditional uniform random variates `curvs` for
        `_make_dependent`.

        Parameters
        ----------
        urvs : array_like
            Uniform random variates to be made dependent by
            `_make_dependent`.
        curvs : array_like
            Array to be filled with dependent conditional uniform random
            variates by `_make_dependent`.

        Returns
        -------
        urvs : array_like
            Dependent uniform random variates.
        curvs : array_like
            Conditional uniform random variates.
        """
        urvs, curvs = self._make_dependent(urvs, curvs)
        if self.is_marginal_layer():
            first_marginal_index = self.output_layer.input_indices[0][0]
            curvs[:, first_marginal_index] = urvs[:, first_marginal_index]
        else:
            copula_index = 0
            curv_index = self.input_marginal_indices[copula_index][1]
            curvs[:, curv_index] = self.curv_ccdf(urvs[:, curv_index],
                                                  curvs, copula_index)
        return urvs, curvs

    def curv_ccdf(self, sample, curvs, copula_index):
        """Helper function for `build_curvs`.

        The function generates a conditional sample.

        Parameters
        ----------
        sample : float
            Right input for the marginal layer.
        curvs : array_like
            Conditional uniform random variates.
        copula_index : int
            Index of the copula to be used to generate the dependent
            sample.

        Returns
        -------
        sample : float
            Conditional sample for `curvs` at index `copula_index`.
        """
        if not self.is_marginal_layer():
            sample = self.input_layer.curv_ccdf(
                sample, curvs, self.input_indices[copula_index][1])
            curv_index = self.input_marginal_indices[copula_index][0]
            input_urvs = np.array([curvs[:, curv_index], sample]).T
            sample = self.copulas[copula_index].ccdf(input_urvs, axis=0)
        return sample

    def _make_dependent(self, urvs, curvs=None):
        """Helper function for `rvs`.

        Introduces dependencies between the uniform random variates `urvs`
        according to the vine copula tree.

        Parameters
        ----------
        urvs : array_like
            Uniform random variates to be made dependent.
        curvs : array_like, optional
            Array to be filled with dependent conditional uniform random
            variates by `build_curvs`.  (Default: `None`)

        Returns
        -------
        urvs : array_like
            Dependent uniform random variates.
        curvs : array_like
            Conditional uniform random variates.
        """
        if curvs is None:
            curvs = np.zeros(shape=urvs.shape)
        if not self.is_marginal_layer():
            urvs, curvs = self.input_layer.build_curvs(urvs, curvs)
        copula_index = 0
        layer = self
        while not layer.is_marginal_layer():
            imi = layer.input_marginal_indices[copula_index]
            input_urvs = np.array([curvs[:, imi[0]], urvs[:, imi[1]]]).T
            urvs[:, imi[1]] = layer.copulas[copula_index].ppcf(input_urvs,
                                                               axis=0)
            copula_index = layer.input_indices[copula_index][1]
            layer = layer.input_layer
        return urvs, curvs

    def rvs(self, size=1, random_state=None):
        """Generates random variates from the mixed vine.

        Currently assumes a c-vine structure.

        Parameters
        ----------
        size : int, optional
            The number of samples to generate.  (Default: 1)
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            The random state to use for random variate generation.  `None`
            corresponds to the `RandomState` singleton.  For an `int`, a
            new `RandomState` is generated and seeded.  For a
            `RandomState` or `Generator`, the object is used.
            (Default: `None`)

        Returns
        -------
        array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.
        """
        if self.is_root_layer():
            # Determine distribution dimension
            layer = self
            while not layer.is_marginal_layer():
                layer = layer.input_layer
            dim = len(layer.marginals)
            samples = uniform.rvs(size=[size, dim], random_state=random_state)
            samples, _ = self._make_dependent(samples)
            # Use marginals to transform dependent uniform samples
            for i, marginal in enumerate(layer.marginals):
                samples[:, i] = marginal.ppf(samples[:, i])
            return samples
        return self.output_layer.rvs(size=size, random_state=random_state)

    def fit(self, samples, is_continuous, trunc_level=None):
        """Fits the vine tree to the given samples.

        This method is supposed to be called on the output layer and will
        recurse to its input layers.

        Parameters
        ----------
        samples : array_like
            n-by-d matrix of samples where n is the number of samples and d
            is the number of marginals.
        is_continuous : array_like
            List of boolean values of length d, where d is the number of
            marginals and element i is `True` if marginal i is continuous.
        trunc_level : int, optional
            Layer level to truncate the vine at.  Copulas in layers beyond
            are just independence copulas.  If the level is `None`, then
            the vine is not truncated.  (Default: `None`)

        Returns
        -------
        output_urvs : array_like
            The output uniform random variates of the layer.  Can be
            ignored if this is the output layer.
        """
        if self.is_marginal_layer():
            output_urvs = np.zeros(samples.shape)
            for i in range(samples.shape[1]):
                self.marginals[i] = Marginal.fit(samples[:, i],
                                                 is_continuous[i])
                output_urvs[:, i] = self.marginals[i].cdf(samples[:, i])
        else:
            input_urvs = self.input_layer.fit(samples, is_continuous,
                                              trunc_level)
            truncate = trunc_level and \
                samples.shape[1] - len(self.input_indices) > trunc_level - 1
            output_urvs = np.zeros((samples.shape[0], len(self.input_indices)))
            for i, i_ind in enumerate(self.input_indices):
                if truncate:
                    self.copulas[i] = IndependenceCopula()
                else:
                    self.copulas[i] = Copula.fit(input_urvs[:, i_ind])
                output_urvs[:, i] = self.copulas[i].ccdf(input_urvs[:, i_ind])
        return output_urvs

    def get_all_params(self):
        """Constructs an array containing all copula parameters.

        Returns
        -------
        params : list
            A list containing all copula parameter values starting with the
            parameters of the first copula layer and continuing layer by
            layer.
        """
        if self.is_marginal_layer():
            params = []
        else:
            params = self.input_layer.get_all_params()
            for copula in self.copulas:
                if copula.theta is not None:
                    if np.ndim(copula.theta) == 0:
                        params.append(copula.theta)
                    else:
                        params = params + list(copula.theta)
        return params

    def set_all_params(self, params):
        """Sets all copula parameters to the values stored in `params`.

        Parameters
        ----------
        params : list
            A list containing all copula parameter values starting with the
            parameters of the first copula layer and continuing layer by
            layer.
        """
        if not self.is_marginal_layer():
            for copula in reversed(self.copulas):
                if copula.theta is not None:
                    if np.ndim(copula.theta) == 0:
                        param_count = 1
                        copula.theta = params[-1]
                    else:
                        param_count = len(copula.theta)
                        copula.theta[:] = params[-param_count:]
                    params = params[:-param_count]
            self.input_layer.set_all_params(params)

    def get_all_bounds(self):
        """Collects the bounds of all copula parameters.

        Returns
        -------
        bnds : list
            A list of 2-tuples containing all copula parameter bounds
            starting with the parameters of the first copula layer and
            continuing layer by layer.  The first element of tuple i
            denotes the lower bound and the second element denotes the
            upper bound of parameter i.
        """
        if self.is_marginal_layer():
            bnds = []
        else:
            bnds = self.input_layer.get_all_bounds()
            for copula in self.copulas:
                for bnd in copula.theta_bounds():
                    bnds.append(bnd)
        return bnds

    def is_continuous(self):
        """Determines which marginals are continuous.

        Returns
        -------
        vals : array_like
            List of boolean values of length d, where d is the number of
            marginals and element i is `True` if marginal i is continuous.
        """
        if self.is_marginal_layer():
            vals = [marginal.is_continuous for marginal in self.marginals]
            return vals
        return self.input_layer.is_continuous()
