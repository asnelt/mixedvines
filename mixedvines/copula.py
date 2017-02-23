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
from scipy.stats import norm, t, uniform, mvn
from scipy.special import gammaln
import numpy as np


class Copula(object):
    '''
    This class represents a copula.
    '''
    def __init__(self, family, theta=None, rotation=None):
        '''
        Constructs a copula of a given family.
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
        Checks family parameter.
        '''
        families = ['ind', 'gaussian', 'clayton']
        if family not in families:
            raise ValueError("Family '" + family + "' not supported.")

    @staticmethod
    def _check_theta(family, theta):
        '''
        Checks the theta parameter.
        '''
        if family == 'ind' and theta is not None:
            raise ValueError("Independent copula has no parameter.")
        if family == 'gaussian' and (theta < -1 or theta > 1):
            raise ValueError("For Gaussian family, 'theta' must be a scalar in"
                             " [-1, 1].")
        if family == 'clayton' and theta < 0:
            raise ValueError("For Clayton family, theta must be a non-negative"
                             " scalar.")

    @staticmethod
    def _check_rotation(rotation):
        '''
        Checks rotation parameter.
        '''
        rotations = ['90°', '180°', '270°']
        if rotation and rotation not in rotations:
            raise ValueError("Rotation '" + rotation + "' not supported.")

    @staticmethod
    def _crop_input(u):
        '''
        Crops the input to the unit hypercube.
        '''
        u[u < 0] = 0
        u[u > 1] = 1

    def _rotate_input(self, u):
        '''
        Preprocesses the input to account for the copula rotation.
        '''
        if self.rotation == '90°':
            u[:, 1] = 1 - u[:, 1]
        elif self.rotation == '180°':
            u[:, :] = 1 - u[:, :]
        elif self.rotation == '270°':
            u[:, 0] = 1 - u[:, 0]

    def logpdf(self, u):
        '''
        Calculates the log of the probability density function.
        '''
        u = np.copy(np.asarray(u))
        self._crop_input(u)
        self._rotate_input(u)
        inner = np.all(np.bitwise_and(u != 0.0, u != 1.0), axis=1)
        outer = np.invert(inner)
        if self.family == 'ind':
            val = np.zeros(u.shape[0])
        elif self.family == 'gaussian':
            val = np.zeros(u.shape[0])
            x = norm.ppf(u)
            val[inner] = 2 * self.theta * x[inner, 0] * x[inner, 1] \
                - self.theta**2 * (x[inner, 0]**2 + x[inner, 1]**2)
            val[inner] /= 2 * (1 - self.theta**2)
            val[inner] -= np.log(1 - self.theta**2) / 2
            val[outer] = -np.inf
        elif self.family == 'clayton':
            if self.theta == 0:
                val = np.zeros(u.shape[0])
            else:
                val = np.zeros(u.shape[0])
                val[inner] = np.log(1 + self.theta) \
                    + (-1 - self.theta) \
                    * (np.log(u[inner, 0]) + np.log(u[inner, 1])) \
                    + (-1 / self.theta-2) * np.log(u[inner, 0]**(-self.theta)
                                                   + u[inner, 1]**(-self.theta)
                                                   - 1)
                val[outer] = -np.inf
        return val

    def pdf(self, u):
        '''
        Calculates the probability density function.
        '''
        return np.exp(self.logpdf(u))

    def logcdf(self, u):
        '''
        Calculates the log of the cumulative distribution function.
        '''
        u = np.copy(np.asarray(u))
        self._crop_input(u)
        self._rotate_input(u)
        if self.family == 'ind':
            old_settings = np.seterr(divide='ignore')
            val = np.sum(np.log(u), axis=1)
            np.seterr(**old_settings)
        elif self.family == 'gaussian':
            lower = np.full(2, -np.inf)
            upper = norm.ppf(u)
            limit_flags = np.zeros(2)

            def func1d(upper1d):
                '''
                Calculates the multivariate normal cumulative distribution
                function of a single sample.
                '''
                return mvn.mvndst(lower, upper1d, limit_flags, self.theta)[1]

            val = np.apply_along_axis(func1d, -1, upper)
            val = np.log(val)
            val[np.any(u == 0.0, axis=1)] = -np.inf
            val[u[:, 0] == 1.0] = np.log(u[u[:, 0] == 1.0, 1])
            val[u[:, 1] == 1.0] = np.log(u[u[:, 1] == 1.0, 0])
        elif self.family == 'clayton':
            if self.theta == 0:
                val = np.sum(np.log(u), axis=1)
            else:
                old_settings = np.seterr(divide='ignore')
                val = (-1 / self.theta) \
                    * np.log(np.maximum(u[:, 0]**(-self.theta)
                                        + u[:, 1]**(-self.theta) - 1, 0))
                np.seterr(**old_settings)
        # Transform according to rotation, but take _rotate_input into account
        if self.rotation == '90°':
            old_settings = np.seterr(divide='ignore')
            val = np.log(u[:, 0] - np.exp(val))
            np.seterr(**old_settings)
        elif self.rotation == '180°':
            old_settings = np.seterr(divide='ignore')
            val = np.log((1 - u[:, 0]) + (1 - u[:, 1]) - 1.0 + np.exp(val))
            np.seterr(**old_settings)
        elif self.rotation == '270°':
            old_settings = np.seterr(divide='ignore')
            val = np.log(u[:, 1] - np.exp(val))
            np.seterr(**old_settings)
        return val

    def cdf(self, u):
        '''
        Calculates the cumulative distribution function.
        '''
        return np.exp(self.logcdf(u))

    def ccdf(self, u, axis=1):
        '''
        Calculates the conditional cumulative distribution function.
        '''
        u = np.copy(np.asarray(u))
        self._crop_input(u)
        if axis == 0:
            # Temporarily change rotation
            rotation = self.rotation
            try:
                if self.rotation == '90°':
                    self.rotation = '270°'
                elif self.rotation == '270°':
                    self.rotation = '90°'
                val = self.ccdf(u[:, [1, 0]], axis=1)
            finally:
                # Recover original rotation
                self.rotation = rotation
            return val
        elif axis == 1:
            self._rotate_input(u)
            if self.family == 'ind':
                val = u[:, 0]
            elif self.family == 'gaussian':
                x = norm.ppf(u)
                val = norm.cdf((x[:, 0] - self.theta * x[:, 1])
                               / np.sqrt(1 - self.theta**2))
            elif self.family == 'clayton':
                if self.theta == 0:
                    val = u[:, 0]
                else:
                    val = np.zeros(u.shape[0])
                    gtz = np.all(u > 0.0, axis=1)
                    val[gtz] = np.maximum(u[gtz, 1]**(-1 - self.theta)
                                          * (u[gtz, 0]**(-self.theta)
                                             + u[gtz, 1]**(-self.theta) - 1)
                                          ** (-1 - 1 / self.theta), 0)
            if self.rotation == '180°' or self.rotation == '270°':
                val = 1.0 - val
            return val
        else:
            raise ValueError("axis must be in [0, 1].")

    def ppcf(self, u, axis=1):
        '''
        Calculates the inverse of the copula conditional cumulative
        distribution function.
        '''
        u = np.copy(np.asarray(u))
        self._crop_input(u)
        if axis == 0:
            # Temporarily change rotation
            rotation = self.rotation
            try:
                if self.rotation == '90°':
                    self.rotation = '270°'
                elif self.rotation == '270°':
                    self.rotation = '90°'
                val = self.ppcf(u[:, [1, 0]], axis=1)
            finally:
                # Recover original rotation
                self.rotation = rotation
            return val
        elif axis == 1:
            self._rotate_input(u)
            if self.family == 'ind':
                val = u[:, 0]
            elif self.family == 'gaussian':
                x = norm.ppf(u)
                val = norm.cdf(x[:, 0] * np.sqrt(1 - self.theta**2)
                               + self.theta * x[:, 1])
            elif self.family == 'clayton':
                if self.theta == 0:
                    val = u[:, 0]
                else:
                    val = np.zeros(u.shape[0])
                    gtz = np.all(u > 0.0, axis=1)
                    val[gtz] = (1 - u[gtz, 1]**(-self.theta)
                                + (u[gtz, 0] * (u[gtz, 1]**(1 + self.theta)))
                                ** (-self.theta / (1 + self.theta))) \
                        ** (-1 / self.theta)
            if self.rotation == '180°' or self.rotation == '270°':
                val = 1.0 - val
            return val
        else:
            raise ValueError("axis must be in [0, 1].")

    def rvs(self, size=1):
        '''
        Generates random variates from the copula.
        '''
        samples = np.stack((uniform.rvs(size=size), uniform.rvs(size=size)),
                           axis=1)
        samples[:, 0] = self.ppcf(samples)
        return samples

    @staticmethod
    def fit(samples, family, rotation=None):
        '''
        Fits the parameters of the copula to the given samples.
        '''
        Copula._check_family(family)
        Copula._check_rotation(rotation)
        if family == 'ind':
            return Copula(family, theta=None, rotation=rotation)
        elif family == 'gaussian':
            initial_point = (0.0)
            bnds = [(-1.0, 1.0)]
        elif family == 'clayton':
            initial_point = (1.0)
            bnds = [(1e-3, 20)]
        # Optimize copula parameters
        copula = Copula(family, theta=initial_point, rotation=rotation)

        def fun(theta):
            '''
            Calculates the cost of a given theta parameter.
            '''
            return Copula._theta_cost(theta, samples, copula)

        result = minimize(fun, initial_point, method='TNC', bounds=bnds)
        copula.theta = result.x
        return copula

    @staticmethod
    def _theta_cost(theta, samples, copula):
        '''
        Helper function for theta optimization.
        '''
        copula.theta = np.asarray(theta)
        return -np.sum(copula.logpdf(samples))
