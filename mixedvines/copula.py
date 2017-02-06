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
# from scipy.stats import multivariate_t
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
        Copula._check_family(family, rotation)
        Copula._check_theta(family, theta)
        self.family = family
        self.theta = theta
        self.rotation = rotation

    @staticmethod
    def _check_family(family, rotation):
        '''
        Checks all parameters.
        '''
        families = ['ind', 'gaussian', 'student', 'clayton']
        if family not in families:
            raise ValueError("Family '" + family + "' not supported.")
        if rotation:
            raise NotImplementedError
        # rotations = ['90°', '180°', '270°']
        # if rotation and rotation not in rotations:
        #     raise ValueError("Rotation '" + rotation + "' not supported.")

    @staticmethod
    def _check_theta(family, theta):
        if family == 'ind' and theta != None:
            raise ValueError("Independent copula has no parameter.")
        if family == 'gaussian' and (theta < -1 or theta > 1):
            raise ValueError("For Gaussian family, 'theta' must be a scalar in"
                             " [-1, 1].")
        if family == 'student' \
                and (theta[0] < -1 or theta[0] > 1 or theta[1] <= 0):
            raise ValueError("For Student t family, 'theta[0]' must be in"
                             " [-1, 1] and 'theta[1]' must be positive.")
        if family == 'clayton' and theta < 0:
            raise ValueError("For Clayton family, theta must be a non-negative"
                             " scalar.")

    def logpdf(self, u):
        '''
        Calculates the log of the probability density function.
        '''
        u = np.asarray(u)
        u[u < 0] = 0
        u[u > 1] = 1
        if self.family == 'ind':
            val = np.zeros(u.shape[0])
        elif self.family == 'gaussian':
            x = norm.ppf(u)
            val = 2 * self.theta * x[:, 0] * x[:, 1] \
                    - self.theta**2 * (x[:, 0]**2 + x[:, 1]**2)
            val /= 2 * (1 - self.theta**2)
            val -= np.log(1 - self.theta**2) / 2
        elif self.family == 'student':
            x = t.ppf(u, self.theta[1])
            fac1 = gammaln(self.theta[1] / 2 + 1)
            fac2 = -gammaln(self.theta[1] / 2) - np.log(np.pi) \
                    - np.log(self.theta[1]) - np.log(1 - self.theta[0]**2) \
                    / 2 - np.log(t.pdf(x[:, 0], self.theta(1))) \
                    - np.log(t.pdf(x[:, 1], self.theta[1]))
            fac3 = (-(self.theta[1] + 2) / 2) \
                    * np.log(1 + (x[:, 0]**2 + x[:, 1]**2 \
                                  - self.theta[0] * x[:, 0] * x[:, 1]) \
                                 / (self.theta[1] * (1 - self.theta[0]**2)))
            val = fac1 + fac2 + fac3
        elif self.family == 'clayton':
            if self.theta == 0:
                val = np.zeros(u.shape[0])
            else:
                val = np.log(1 + self.theta) \
                        + (-1 - self.theta) * (np.log(u[:, 0]) \
                        + np.log(u[:, 1])) + (-1 / self.theta-2) \
                        * np.log(u[:, 0]**(-self.theta) \
                                 + u[:, 1]**(-self.theta) - 1)
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
        u = np.asarray(u)
        u[u < 0] = 0
        u[u > 1] = 1
        if self.family == 'ind':
            val = np.sum(np.log(u), axis=1)
        elif self.family == 'gaussian':
            lower = np.full(2, -np.inf)
            upper = norm.ppf(u)
            limit_flags = np.zeros(2)
            func1d = lambda upper1d: mvn.mvndst(lower, upper1d, limit_flags, \
                                                self.theta)[1]
            val = np.apply_along_axis(func1d, -1, upper)
            val = np.log(val)
        elif self.family == 'student':
            raise NotImplementedError
            # val = multivariate_t.logcdf(t.ppf(u, self.theta[1]), \
            #         [[1, self.theta[0]], [self.theta[0], 1]], self.theta[1])
        elif self.family == 'clayton':
            if self.theta == 0:
                val = np.sum(np.log(u), axis=1)
            else:
                val = (-1 / self.theta) \
                        * np.log(np.max(u[:, 0]**(-self.theta) \
                                        + u[:, 1]**(-self.theta) - 1, 0))
        return val

    def cdf(self, u):
        '''
        Calculates the cumulative distribution function.
        '''
        return np.exp(self.logcdf(u))

    def ccdf(self, u):
        '''
        Calculates the conditional cumulative distribution function.
        '''
        u = np.asarray(u)
        u[u < 0] = 0
        u[u > 1] = 1
        if self.family == 'ind':
            val = u[:, 0]
        elif self.family == 'gaussian':
            x = norm.ppf(u)
            val = norm.cdf((x[:, 0] - self.theta * x[:, 1]) \
                           / np.sqrt(1 - self.theta**2))
        elif self.family == 'student':
            x = t.ppf(u, self.theta[1])
            val = t.cdf(np.sqrt((self.theta[1] + 1) \
                                / (self.theta[1] + x[:, 1]**2)) \
                        * (x[:, 0] - self.theta[0] * x[:, 1]) \
                          / (np.sqrt(1 - self.theta[0]**2)), self.theta[1] + 1)
        elif self.family == 'clayton':
            if self.theta == 0:
                val = u[:, 0]
            else:
                val = np.max(u[:, 1]**(-1 - self.theta) \
                             * (u[:, 0]**(-self.theta) \
                                + u[:, 1]**(-self.theta) - 1) \
                               **(-1 - 1 / self.theta), 0)
        return val

    def ppcf(self, u):
        '''
        Calculates the inverse of the copula conditional cumulative distribution
        function.
        '''
        u = np.asarray(u)
        u[u < 0] = 0
        u[u > 1] = 1
        if self.family == 'ind':
            val = u[:, 0]
        elif self.family == 'gaussian':
            x = norm.ppf(u)
            val = norm.cdf(x[:, 0] * np.sqrt(1 - self.theta**2) \
                           + self.theta * x[:, 1])
        elif self.family == 'student':
            x = t.ppf(u, self.theta[1])
            val = t.cdf(np.sqrt(((1 - self.theta[0]**2) \
                                 * (self.theta[1] + x[:, 1]**2)) \
                                / (self.theta[1] + 1)) \
                        * t.ppf(u[:, 0], self.theta[1] + 1) \
                        + self.theta[0] * x[:, 1], self.theta[1])
        elif self.family == 'clayton':
            if self.theta == 0:
                val = u[:, 0]
            else:
                val = (1 - u[:, 1]**(-self.theta) \
                       + (u[:, 0] * (u[:, 1]**(1 + self.theta))) \
                         **(-self.theta / (1 + self.theta)))**(-1 / self.theta)
        return val

    def rvs(self, size=1):
        '''
        Generates random variates from the copula.
        '''
        samples = np.stack((uniform.rvs(size=size), uniform.rvs(size=size)), \
                           axis=1)
        samples[:, 0] = self.ppcf(samples)
        return samples

    @staticmethod
    def fit(samples, family, rotation=None):
        '''
        Fits the parameters of the copula to the given samples.
        '''
        samples = np.asarray(samples)
        samples[samples < 0] = 0
        samples[samples > 1] = 1
        Copula._check_family(family, rotation)
        if family == 'ind':
            return Copula(family, theta=None, rotation=rotation)
        elif family == 'gaussian':
            initial_point = (0.0)
            bnds = [(-1.0, 1.0)]
        elif family == 'student':
            initial_point = (0.0, 1.0)
            bnds = [(-1.0, 1.0), (1e-1, 1000)]
        elif family == 'clayton':
            initial_point = (1.0)
            bnds = [(1e-3, 20)]
        # Optimize copula parameters
        copula = Copula(family, theta=initial_point, rotation=rotation)
        fun = lambda theta: Copula._theta_cost(theta, samples, copula)
        result = minimize(fun, initial_point, method='TNC', bounds=bnds)
        copula.theta = result.x
        return copula

    @staticmethod
    def _theta_cost(theta, samples, copula):
        '''
        Helper function for theta optimization
        '''
        copula.theta = np.asarray(theta)
        return -np.sum(copula.logpdf(samples))
