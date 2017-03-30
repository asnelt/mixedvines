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
import sys
import abc
from scipy.optimize import minimize
from scipy.stats import norm, uniform, mvn
import numpy as np


# Ensure abstract base class compatibility
if sys.version_info[0] == 3 and sys.version_info[1] >= 4 \
        or sys.version_info[0] > 3:
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Copula(ABC):
    '''
    This abstract class represents a copula.

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
    ``estimate_theta(samples)``
        Estimates the `theta` parameters from the given samples.
    ``fit(samples)``
        Fit a copula to samples.
    ``theta_bounds()``
        Bounds for `theta` parameters.
    '''
    rotation_options = ['90°', '180°', '270°']

    def __init__(self, theta=None, rotation=None):
        '''
        Constructs a copula of a given family.

        Parameters
        ----------
        theta : array_like
            Parameter array of the copula.  The number of elements depends on
            the copula family.
        rotation : string, optional
            Clockwise rotation of the copula.  Can be one of the elements of
            `Copula.rotation_options` or `None`.  (Default: None)
        '''
        Copula._check_theta(theta)
        Copula._check_rotation(rotation)
        self.theta = theta
        self.rotation = rotation

    @staticmethod
    @abc.abstractmethod
    def _check_theta(theta):
        '''
        Checks the `theta` parameter.

        Parameters
        ----------
        theta : array_like
            Parameter array of the copula.  The number of elements depends on
            the copula family.
        '''
        pass

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

    @abc.abstractmethod
    def _logpdf(self, samples):
        '''
        Calculates the log of the probability density function. The samples can
        be assumed to lie within the unit hypercube.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Log of the probability density function evaluated at `samples`.
        '''
        pass

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
        vals[inner] = self._logpdf(samples[inner, :])
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

    @abc.abstractmethod
    def _logcdf(self, samples):
        '''
        Calculates the log of the cumulative distribution function. The samples
        can be assumed to lie within the unit hypercube.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Log of the cumulative distribution function evaluated at `samples`.
        '''
        pass

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
        vals = self._logcdf(samples)
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

    @abc.abstractmethod
    def _ccdf(self, samples):
        '''
        Calculates the conditional cumulative distribution function conditioned
        on axis 1. The samples can be assumed to lie within the unit hypercube.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Conditional cumulative distribution function evaluated at
            `samples`.
        TODO.
        '''
        pass

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
            vals = self._ccdf(samples)
            if self.rotation == '180°' or self.rotation == '270°':
                vals = 1.0 - vals
        finally:
            # Recover original rotation
            self.rotation = rotation
        return vals

    @abc.abstractmethod
    def _ppcf(self, samples):
        '''
        Calculates the inverse of the copula conditional cumulative
        distribution function conditioned on axis 1. The samples can be assumed
        to lie within the unit hypercube.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Inverse of the conditional cumulative distribution function
            evaluated at `samples`.
        '''
        pass

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
            # Temporarily change rotation according to axis
            samples = self._axis_rotate(samples, axis)
            samples = self._rotate_input(samples)
            vals = self._ppcf(samples)
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

    def estimate_theta(self, samples):
        '''
        Estimates the theta parameters from the given samples.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        '''
        if self.theta:
            bnds = self.theta_bounds()

            def cost(theta):
                '''
                Calculates the cost of a given `theta` parameter.
                '''
                self.theta = np.asarray(theta)
                return -np.sum(self.logpdf(samples))

            result = minimize(cost, self.theta, method='TNC', bounds=bnds)
            self.theta = result.x

    @staticmethod
    def fit(samples):
        '''
        Fits the parameters of the copula to the given samples.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        copula : Copula
            The copula fitted to `samples`.
        '''
        # Find best fitting family
        copulas = []
        for family in Copula.__subclasses__():
            copulas.append(family.fit(samples))
        # Calculate Akaike information criterion
        aic = np.zeros(len(copulas))
        for i, copula in enumerate(copulas):
            aic[i] = - 2 * np.sum(copula.logpdf(samples))
            if copula.theta:
                aic[i] += 2 * len(copula.theta)
        copula = copulas[np.argmin(aic)]
        return copula

    @staticmethod
    @abc.abstractmethod
    def theta_bounds():
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
        pass


class IndependenceCopula(Copula):
    '''
    This class represents the independence copula.
    '''

    @staticmethod
    def _check_theta(theta):
        if theta is not None:
            raise ValueError("Independence copula has no parameter.")

    def _logpdf(self, samples):
        vals = np.zeros(samples.shape[0])
        return vals

    def _logcdf(self, samples):
        old_settings = np.seterr(divide='ignore')
        vals = np.sum(np.log(samples), axis=1)
        np.seterr(**old_settings)
        return vals

    def _ccdf(self, samples):
        vals = samples[:, 0]
        return vals

    def _ppcf(self, samples):
        vals = samples[:, 0]
        return vals

    @staticmethod
    def fit(samples):
        copula = IndependenceCopula()
        return copula

    @staticmethod
    def theta_bounds():
        bnds = []
        return bnds


class GaussianCopula(Copula):
    '''
    This class represents a copula from the Gaussian family.
    '''

    @staticmethod
    def _check_theta(theta):
        if theta < -1 or theta > 1:
            raise ValueError("For Gaussian family, 'theta' must be a scalar in"
                             " [-1, 1].")

    def _logpdf(self, samples):
        if self.theta >= 1.0:
            vals = np.zeros(samples.shape[0])
            vals[samples[:, 0] == samples[:, 1]] = np.inf
        elif self.theta <= -1.0:
            vals = np.zeros(samples.shape[0])
            vals[samples[:, 0] == 1 - samples[:, 1]] = np.inf
        else:
            nrvs = norm.ppf(samples)
            vals = 2 * self.theta * nrvs[:, 0] * nrvs[:, 1] - self.theta**2 \
                * (nrvs[:, 0]**2 + nrvs[:, 1]**2)
            vals /= 2 * (1 - self.theta**2)
            vals -= np.log(1 - self.theta**2) / 2
        return vals

    def _logcdf(self, samples):
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
        vals[samples[:, 0] == 1.0] = np.log(samples[samples[:, 0] == 1.0, 1])
        vals[samples[:, 1] == 1.0] = np.log(samples[samples[:, 1] == 1.0, 0])
        return vals

    def _ccdf(self, samples):
        nrvs = norm.ppf(samples)
        vals = norm.cdf((nrvs[:, 0] - self.theta * nrvs[:, 1])
                        / np.sqrt(1 - self.theta**2))
        return vals

    def _ppcf(self, samples):
        nrvs = norm.ppf(samples)
        vals = norm.cdf(nrvs[:, 0] * np.sqrt(1 - self.theta**2)
                        + self.theta * nrvs[:, 1])
        return vals

    @staticmethod
    def fit(samples):
        initial_point = (0.0)
        copula = GaussianCopula(theta=initial_point)
        copula.estimate_theta(samples)
        return copula

    @staticmethod
    def theta_bounds():
        bnds = [(-1.0 + 1e-3, 1.0 - 1e-3)]
        return bnds


class ClaytonCopula(Copula):
    '''
    This class represents a copula from the Clayton family.
    '''

    @staticmethod
    def _check_theta(theta):
        if theta < 0:
            raise ValueError("For Clayton family, theta must be a non-negative"
                             " scalar.")

    def _logpdf(self, samples):
        if self.theta == 0:
            vals = np.zeros(samples.shape[0])
        else:
            vals = np.log(1 + self.theta) + (-1 - self.theta) \
                   * (np.log(samples[:, 0]) + np.log(samples[:, 1])) \
                   + (-1 / self.theta-2) \
                   * np.log(samples[:, 0]**(-self.theta)
                            + samples[:, 1]**(-self.theta) - 1)
        return vals

    def _logcdf(self, samples):
        if self.theta == 0:
            vals = np.sum(np.log(samples), axis=1)
        else:
            old_settings = np.seterr(divide='ignore')
            vals = (-1 / self.theta) \
                * np.log(np.maximum(samples[:, 0]**(-self.theta)
                                    + samples[:, 1]**(-self.theta) - 1, 0))
            np.seterr(**old_settings)
        return vals

    def _ccdf(self, samples):
        if self.theta == 0:
            vals = samples[:, 0]
        else:
            vals = np.zeros(samples.shape[0])
            gtz = np.all(samples > 0.0, axis=1)
            vals[gtz] = np.maximum(samples[gtz, 1]**(-1 - self.theta)
                                   * (samples[gtz, 0]**(-self.theta)
                                      + samples[gtz, 1]**(-self.theta) - 1)
                                   ** (-1 - 1 / self.theta), 0)
        return vals

    def _ppcf(self, samples):
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
        return vals

    @staticmethod
    def fit(samples):
        initial_point = (1.0)
        # Optimize rotation as well
        copulas = [ClaytonCopula(theta=initial_point)]
        for rotation in Copula.rotation_options:
            copulas.append(ClaytonCopula(theta=initial_point,
                                         rotation=rotation))
        # Fit parameters and calculate Akaike information criterion
        aic = np.zeros(len(copulas))
        for i, _ in enumerate(copulas):
            copulas[i].estimate_theta(samples)
            aic[i] = - 2 * np.sum(copulas[i].logpdf(samples)) \
                + 2 * len(copulas[i].theta)
        # Select best copula
        copula = copulas[np.argmin(aic)]
        return copula

    @staticmethod
    def theta_bounds():
        bnds = [(1e-3, 20)]
        return bnds


class FrankCopula(Copula):
    '''
    This class represents a copula from the Frank family.
    '''

    @staticmethod
    def _check_theta(theta):
        if theta is None:
            raise ValueError("For Frank family, theta must be a scalar.")

    def _logpdf(self, samples):
        if self.theta == 0:
            vals = np.zeros(samples.shape[0])
        else:
            vals = np.log(-self.theta * np.expm1(-self.theta)
                          * np.exp(-self.theta
                                   * (samples[:, 0] + samples[:, 1]))
                          / (np.expm1(-self.theta)
                             + np.expm1(-self.theta * samples[:, 0])
                             * np.expm1(-self.theta * samples[:, 1])) ** 2)
        return vals

    def _logcdf(self, samples):
        if self.theta == 0:
            vals = np.sum(np.log(samples), axis=1)
        else:
            old_settings = np.seterr(divide='ignore')
            vals = np.log(-np.log1p(np.expm1(-self.theta * samples[:, 0])
                                    * np.expm1(-self.theta * samples[:, 1])
                                    / (np.expm1(-self.theta)))) \
                - np.log(self.theta)
            np.seterr(**old_settings)
        return vals

    def _ccdf(self, samples):
        if self.theta == 0:
            vals = samples[:, 0]
        else:
            vals = np.exp(-self.theta * samples[:, 1]) \
                * np.expm1(-self.theta * samples[:, 0]) \
                / (np.expm1(-self.theta)
                   + np.expm1(-self.theta * samples[:, 0])
                   * np.expm1(-self.theta * samples[:, 1]))
        return vals

    def _ppcf(self, samples):
        if self.theta == 0:
            vals = samples[:, 0]
        else:
            vals = -np.log1p(samples[:, 0] * np.expm1(-self.theta)
                             / (np.exp(-self.theta * samples[:, 1])
                                - samples[:, 0] * np.expm1(-self.theta
                                                           * samples[:, 1]))) \
                / self.theta
        return vals

    @staticmethod
    def fit(samples):
        initial_point = (0.0)
        copula = FrankCopula(theta=initial_point)
        copula.estimate_theta(samples)
        return copula

    @staticmethod
    def theta_bounds():
        bnds = [(-20, 20)]
        return bnds
