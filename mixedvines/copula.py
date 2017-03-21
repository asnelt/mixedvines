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
This module implements copula distributions.
'''
from __future__ import division
from scipy.optimize import minimize
from scipy.stats import norm, uniform, mvn
import numpy as np


class Copula(object):
    '''
    This class represents a copula.

    Methods
    -------
    ``logpdf(samples)``
        Log of the probability density function.
    ``pdf(samples)``
        Probability density function.
    ``logcdf(samples)``
        Log of the cumulative distribution function.
    ``cdf(samples)``
        Cumulative distribution function.
    ``ccdf(samples, axis=1)``
        Conditional cumulative distribution function.
    ``ppcf(samples, axis=1)``
        Inverse of the conditional cumulative distribution function.
    ``rvs(size=1)``
        Generate random variates.
    ``fit(samples, family=None, rotation=None)``
        Fit a copula to samples.
    ``theta_bounds(family)``
        Bounds for `theta` parameters.
    '''
    family_options = ['ind', 'gaussian', 'clayton', 'frank']
    rotation_options = ['90°', '180°', '270°']

    def __init__(self, family, theta=None, rotation=None):
        '''
        Constructs a copula of a given family.

        Parameters
        ----------
        family : string
            Family name of the copula.  Can be one of the elements of
            `Copula.family_options`.
        theta : array_like
            Parameter array of the copula.  The number of elements depends on
            the copula family.
        rotation : string, optional
            Rotation of the copula.  Can be one of the elements of
            `Copula.rotation_options` or `None`.  (Default: None)
        '''
        Copula._check_family(family)
        Copula._check_theta(family, theta)
        Copula._check_rotation(rotation)
        self.family = family
        self.theta = theta
        self.rotation = rotation

    @staticmethod
    def _check_family(family):
        '''
        Checks the `family` parameter.

        Parameters
        ----------
        family : string
            Family name of the copula.  Can be one of the elements of
            `Copula.family_options`.
        '''
        if family not in Copula.family_options:
            raise ValueError("Family '" + family + "' not supported.")

    @staticmethod
    def _check_theta(family, theta):
        '''
        Checks the `theta` parameter.

        Parameters
        ----------
        theta : array_like
            Parameter array of the copula.  The number of elements depends on
            the copula family.
        '''
        if family == 'ind' and theta is not None:
            raise ValueError("Independent copula has no parameter.")
        if family == 'gaussian' and (theta < -1 or theta > 1):
            raise ValueError("For Gaussian family, 'theta' must be a scalar in"
                             " [-1, 1].")
        if family == 'clayton' and theta < 0:
            raise ValueError("For Clayton family, theta must be a non-negative"
                             " scalar.")
        if family == 'frank' and theta is None:
            raise ValueError("For Frank family, theta must be a scalar.")

    @staticmethod
    def _check_rotation(rotation):
        '''
        Checks the `rotation` parameter.

        Parameters
        ----------
        rotation : string
            Rotation of the copula.  Can be one of the elements of
            `Copula.rotation_options` or `None`.
        '''
        if rotation and rotation not in Copula.rotation_options:
            raise ValueError("Rotation '" + rotation + "' not supported.")

    @staticmethod
    def _crop_input(samples):
        '''
        Crops the input to the unit hypercube.  The input is changed and a
        reference to the input is returned.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        samples : array_like
            n-by-2 matrix of cropped samples where n is the number of samples.
        '''
        samples[samples < 0] = 0
        samples[samples > 1] = 1
        return samples

    def _rotate_input(self, samples):
        '''
        Preprocesses the input to account for the copula rotation.  The input
        is changed and a reference to the input is returned.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        samples : array_like
            n-by-2 matrix of rotated samples where n is the number of samples.
        '''
        if self.rotation == '90°':
            samples[:, 1] = 1 - samples[:, 1]
        elif self.rotation == '180°':
            samples[:, :] = 1 - samples[:, :]
        elif self.rotation == '270°':
            samples[:, 0] = 1 - samples[:, 0]
        return samples

    def _axis_rotate(self, samples, axis):
        '''
        Changes rotation and samples such that `axis == 0` corresponds to
        `axis == 1`.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        axis : integer
            The axis to condition the cumulative distribution function on.

        Returns
        -------
        samples : array_like
            n-by-2 matrix of rotated samples where n is the number of samples.
        '''
        if axis == 0:
            if self.rotation == '90°':
                self.rotation = '270°'
            elif self.rotation == '270°':
                self.rotation = '90°'
            samples = samples[:, [1, 0]]
        elif axis != 1:
            raise ValueError("axis must be in [0, 1].")
        return samples

    def logpdf(self, samples):
        '''
        Calculates the log of the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Log of the probability density function evaluated at `samples`.
        '''
        samples = np.copy(np.asarray(samples))
        samples = self._crop_input(samples)
        samples = self._rotate_input(samples)
        inner = np.all(np.bitwise_and(samples != 0.0, samples != 1.0), axis=1)
        outer = np.invert(inner)
        vals = np.zeros(samples.shape[0])
        # For 'ind' family, val remains at zero
        if self.family == 'gaussian':
            if self.theta >= 1.0:
                vals[np.bitwise_and(inner,
                                    samples[:, 0] == samples[:, 1])] = np.inf
            elif self.theta <= -1.0:
                vals[np.bitwise_and(inner,
                                    samples[:, 0] == 1 - samples[:, 1])] \
                    = np.inf
            else:
                nrvs = norm.ppf(samples)
                vals[inner] = 2 * self.theta * nrvs[inner, 0] \
                    * nrvs[inner, 1] - self.theta**2 \
                    * (nrvs[inner, 0]**2 + nrvs[inner, 1]**2)
                vals[inner] /= 2 * (1 - self.theta**2)
                vals[inner] -= np.log(1 - self.theta**2) / 2
        elif self.family == 'clayton':
            if self.theta != 0:
                vals[inner] = np.log(1 + self.theta) \
                    + (-1 - self.theta) \
                    * (np.log(samples[inner, 0])
                       + np.log(samples[inner, 1])) \
                    + (-1 / self.theta-2) \
                    * np.log(samples[inner, 0]**(-self.theta)
                             + samples[inner, 1]**(-self.theta) - 1)
        elif self.family == 'frank':
            if self.theta != 0:
                vals[inner] = np.log(-self.theta * np.expm1(-self.theta)
                                     * np.exp(-self.theta
                                              * (samples[inner, 0]
                                                 + samples[inner, 1]))
                                     / (np.expm1(-self.theta)
                                        + np.expm1(-self.theta
                                                   * samples[inner, 0])
                                        * np.expm1(-self.theta
                                                   * samples[inner, 1])) ** 2)
        # Assign zero mass to border
        vals[outer] = -np.inf
        return vals

    def pdf(self, samples):
        '''
        Calculates the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

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
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Log of the cumulative distribution function evaluated at `samples`.
        '''
        samples = np.copy(np.asarray(samples))
        samples = self._crop_input(samples)
        samples = self._rotate_input(samples)
        if self.family == 'ind':
            old_settings = np.seterr(divide='ignore')
            vals = np.sum(np.log(samples), axis=1)
            np.seterr(**old_settings)
        elif self.family == 'gaussian':
            lower = np.full(2, -np.inf)
            upper = norm.ppf(samples)
            limit_flags = np.zeros(2)

            def func1d(upper1d):
                '''
                Calculates the multivariate normal cumulative distribution
                function of a single sample.
                '''
                return mvn.mvndst(lower, upper1d, limit_flags, self.theta)[1]

            vals = np.apply_along_axis(func1d, -1, upper)
            vals = np.log(vals)
            vals[np.any(samples == 0.0, axis=1)] = -np.inf
            vals[samples[:, 0] == 1.0] \
                = np.log(samples[samples[:, 0] == 1.0, 1])
            vals[samples[:, 1] == 1.0] \
                = np.log(samples[samples[:, 1] == 1.0, 0])
        elif self.family == 'clayton':
            if self.theta == 0:
                vals = np.sum(np.log(samples), axis=1)
            else:
                old_settings = np.seterr(divide='ignore')
                vals = (-1 / self.theta) \
                    * np.log(np.maximum(samples[:, 0]**(-self.theta)
                                        + samples[:, 1]**(-self.theta) - 1,
                                        0))
                np.seterr(**old_settings)
        elif self.family == 'frank':
            if self.theta == 0:
                vals = np.sum(np.log(samples), axis=1)
            else:
                old_settings = np.seterr(divide='ignore')
                vals = np.log(-np.log1p(np.expm1(-self.theta * samples[:, 0])
                                        * np.expm1(-self.theta * samples[:, 1])
                                        / (np.expm1(-self.theta)))) \
                    - np.log(self.theta)
                np.seterr(**old_settings)
        # Transform according to rotation, but take `_rotate_input` into
        # account.
        if self.rotation == '90°':
            old_settings = np.seterr(divide='ignore')
            vals = np.log(samples[:, 0] - np.exp(vals))
            np.seterr(**old_settings)
        elif self.rotation == '180°':
            old_settings = np.seterr(divide='ignore')
            vals = np.log((1 - samples[:, 0]) + (1 - samples[:, 1]) - 1.0
                          + np.exp(vals))
            np.seterr(**old_settings)
        elif self.rotation == '270°':
            old_settings = np.seterr(divide='ignore')
            vals = np.log(samples[:, 1] - np.exp(vals))
            np.seterr(**old_settings)
        return vals

    def cdf(self, samples):
        '''
        Calculates the cumulative distribution function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Cumulative distribution function evaluated at `samples`.
        '''
        return np.exp(self.logcdf(samples))

    def ccdf(self, samples, axis=1):
        '''
        Calculates the conditional cumulative distribution function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        axis : integer, optional
            The axis to condition the cumulative distribution function on.
            (Default: 1)

        Returns
        -------
        vals : ndarray
            Conditional cumulative distribution function evaluated at
            `samples`.
        '''
        samples = np.copy(np.asarray(samples))
        samples = self._crop_input(samples)
        rotation = self.rotation
        try:
            # Temporarily change rotation according to axis
            samples = self._axis_rotate(samples, axis)
            samples = self._rotate_input(samples)
            if self.family == 'ind':
                vals = samples[:, 0]
            elif self.family == 'gaussian':
                nrvs = norm.ppf(samples)
                vals = norm.cdf((nrvs[:, 0] - self.theta * nrvs[:, 1])
                                / np.sqrt(1 - self.theta**2))
            elif self.family == 'clayton':
                if self.theta == 0:
                    vals = samples[:, 0]
                else:
                    vals = np.zeros(samples.shape[0])
                    gtz = np.all(samples > 0.0, axis=1)
                    vals[gtz] = np.maximum(samples[gtz, 1]**(-1 - self.theta)
                                           * (samples[gtz, 0]**(-self.theta)
                                              + samples[gtz, 1]**(-self.theta)
                                              - 1)
                                           ** (-1 - 1 / self.theta), 0)
            elif self.family == 'frank':
                if self.theta == 0:
                    vals = samples[:, 0]
                else:
                    vals = np.exp(-self.theta * samples[:, 1]) \
                        * np.expm1(-self.theta * samples[:, 0]) \
                        / (np.expm1(-self.theta)
                           + np.expm1(-self.theta * samples[:, 0])
                           * np.expm1(-self.theta * samples[:, 1]))
            if self.rotation == '180°' or self.rotation == '270°':
                vals = 1.0 - vals
        finally:
            # Recover original rotation
            self.rotation = rotation
        return vals

    def ppcf(self, samples, axis=1):
        '''
        Calculates the inverse of the copula conditional cumulative
        distribution function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        axis : integer, optional
            The axis to condition the cumulative distribution function on.
            (Default: 1)

        Returns
        -------
        vals : ndarray
            Inverse of the conditional cumulative distribution function
            evaluated at `samples`.
        '''
        samples = np.copy(np.asarray(samples))
        samples = self._crop_input(samples)
        rotation = self.rotation
        try:
            # Temporarily change rotation
            samples = self._axis_rotate(samples, axis)
            samples = self._rotate_input(samples)
            if self.family == 'ind':
                vals = samples[:, 0]
            elif self.family == 'gaussian':
                nrvs = norm.ppf(samples)
                vals = norm.cdf(nrvs[:, 0] * np.sqrt(1 - self.theta**2)
                                + self.theta * nrvs[:, 1])
            elif self.family == 'clayton':
                if self.theta == 0:
                    vals = samples[:, 0]
                else:
                    vals = np.zeros(samples.shape[0])
                    gtz = np.all(samples > 0.0, axis=1)
                    vals[gtz] = (1 - samples[gtz, 1]**(-self.theta)
                                 + (samples[gtz, 0]
                                    * (samples[gtz, 1]**(1 + self.theta)))
                                 ** (-self.theta / (1 + self.theta))) \
                        ** (-1 / self.theta)
            elif self.family == 'frank':
                if self.theta == 0:
                    vals = samples[:, 0]
                else:
                    vals = -np.log1p(samples[:, 0] * np.expm1(-self.theta)
                                     / (np.exp(-self.theta * samples[:, 1])
                                        - samples[:, 0]
                                        * np.expm1(-self.theta
                                                   * samples[:, 1]))) \
                        / self.theta
            if self.rotation == '180°' or self.rotation == '270°':
                vals = 1.0 - vals
        finally:
            # Recover orginial rotation
            self.rotation = rotation
        return vals

    def rvs(self, size=1):
        '''
        Generates random variates from the copula.

        Parameters
        ----------
        size : integer, optional
            The number of samples to generate.  (Default: 1)

        Returns
        -------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        '''
        samples = np.stack((uniform.rvs(size=size), uniform.rvs(size=size)),
                           axis=1)
        samples[:, 0] = self.ppcf(samples)
        return samples

    @staticmethod
    def fit(samples, family=None, rotation=None):
        '''
        Fits the parameters of the copula to the given samples.

        If `family` is `None` then the best fitting family is automatically
        selected based on the Akaike information criterion.  For the Clayton
        family, also the rotation is selected automatically if `family` and
        `rotation` are both `None`.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        family : string, optional
            Family name of the copula.  Can be one of the elements of
            `Copula.family_options`.  (Default: best fitting family name)
        rotation : string, optional
            Rotation of the copula.  Can be one of the elements of
            `Copula.rotation_options` or `None`.  (Default: None)

        Returns
        -------
        copula : Copula
            The copula fitted to `samples`.
        '''
        if family:
            Copula._check_family(family)
            Copula._check_rotation(rotation)
            if family == 'ind':
                return Copula(family, theta=None, rotation=rotation)
            elif family == 'gaussian':
                initial_point = (0.0)
            elif family == 'clayton':
                initial_point = (1.0)
            elif family == 'frank':
                initial_point = (0.0)
            # Optimize copula parameters
            bnds = Copula.theta_bounds(family)
            copula = Copula(family, theta=initial_point, rotation=rotation)

            def cost(theta):
                '''
                Calculates the cost of a given `theta` parameter.
                '''
                return Copula._theta_cost(theta, samples, copula)

            result = minimize(cost, initial_point, method='TNC', bounds=bnds)
            copula.theta = result.x
        else:
            # Also find best fitting family
            copulas = []
            for family in Copula.family_options:
                copulas.append(Copula.fit(samples, family))
                # For Clayton family, optimize rotation as well
                if family == 'clayton':
                    for rotation in Copula.rotation_options:
                        copulas.append(Copula.fit(samples, family, rotation))
            # Calculate Akaike information criterion
            aic = np.zeros(len(copulas))
            for i, copula in enumerate(copulas):
                aic[i] = - 2 * np.sum(copula.logpdf(samples))
                if copula.theta:
                    aic[i] += 2 * len(copula.theta)
            copula = copulas[np.argmin(aic)]
        return copula

    @staticmethod
    def theta_bounds(family):
        '''
        Bounds for `theta` parameters.

        Parameters
        ----------
        family : string
            Family name of the copula.  Can be one of the elements of
            `Copula.family_options`.

        Returns
        -------
        bnds : array_like
            n-by-2 matrix of bounds where the first column represents the lower
            bounds and the second column represents the upper bounds.
        '''
        if family == 'gaussian':
            bnds = [(-1.0 + 1e-3, 1.0 - 1e-3)]
        elif family == 'clayton':
            bnds = [(1e-3, 20)]
        elif family == 'frank':
            bnds = [(-20, 20)]
        else:
            bnds = []
        return bnds

    @staticmethod
    def _theta_cost(theta, samples, copula):
        '''
        Helper function for `theta` optimization.

        Parameters
        ----------
        theta : array_like
            Parameter array of the copula.  The number of elements depends on
            the copula family.
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        copula : Copula
            The copula to optimize.

        Returns
        -------
        val : float
            The cost of a particular parameter vector `theta`.
        '''
        copula.theta = np.asarray(theta)
        return -np.sum(copula.logpdf(samples))
