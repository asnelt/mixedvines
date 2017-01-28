# Copyright (C) 2017 Arno Onken
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
'''
This module implements copula-based distributions.
'''
from __future__ import division
from math import pi
from scipy.stats import norm, t, multivariate_normal, multivariate_t
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
        families = ['ind', 'gaussian', 'student', 'clayton']
        if family not in families:
            raise ValueError("Family '" + family + "' not supported.")
        if family == 'ind' and theta != None:
            raise ValueError("Independent copula has no parameter.")
        rotations = ['90°', '180°', '270°']
        if rotation and rotation not in rotations:
            raise ValueError("Rotation '" + rotation + "' not supported.")
        self.family = family
        self.theta = theta
        self.rotation = rotation

    def logpdf(self, u):
        '''
        Calculates the log of the probability density function.
        '''
        if self.family == 'ind':
            val = 0.0
        elif self.family == 'gaussian':
            x = norm.ppf(u)
            val = 2 * self.theta * x[:, 0] * x[:, 1] - self.theta**2 * (x[:, 0]**2 + x[:, 1]**2)
            val /= 2 * (1 - self.theta**2)
            val -= np.log(1 - self.theta**2) / 2
        elif self.family == 'student':
            x = t.ppf(u, self.theta[1])
            fac1 = gammaln(self.theta[1] / 2 + 1)
            fac2 = -gammaln(self.theta[1] / 2) - np.log(pi) - np.log(self.theta[1]) \
                    - np.log(1 - self.theta[0]**2) / 2 - np.log(t.pdf(x[:, 0], self.theta(1))) \
                    - np.log(t.pdf(x[:, 1], self.theta[1]))
            fac3 = (-(self.theta[1] + 2) / 2) * np.log(1 \
                    + (x[:, 0]**2 + x[:, 1]**2 - self.theta[0] * x[:, 0] * x[:, 1]) \
                    / (self.theta[1] * (1 - self.theta[0]**2)))
            val = fac1 + fac2 + fac3
        elif self.family == 'clayton':
            if self.theta == 0:
                val = 0.0
            else:
                val = np.log(1 + self.theta) \
                        + (-1 - self.theta) * (np.log(u[:, 0]) + np.log(u[:, 1])) \
                        + (-1 / self.theta-2) \
                        * np.log(u[:, 0]**(-self.theta) + u[:, 1]**(-self.theta) - 1)
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
        if self.family == 'ind':
            val = np.sum(np.log(u), axis=1)
        elif self.family == 'gaussian':
            val = multivariate_normal.logcdf(norm.ppf(u), cov=[[1, self.theta], [self.theta, 1]])
        elif self.family == 'student':
            val = multivariate_t.logcdf(t.ppf(u, self.theta[1]), \
                    [[1, self.theta[0]], [self.theta[0], 1]], self.theta[1])
        elif self.family == 'clayton':
            if self.theta == 0:
                val = np.sum(np.log(u), axis=1)
            else:
                val = (-1 / self.theta) \
                        * np.log(np.max(u[:, 0]**(-self.theta) + u[:, 1]**(-self.theta) - 1, 0))
        return val

    def cdf(self, u):
        '''
        Calculates the cumulative distribution function.
        '''
        return np.exp(self.logcdf(u))

    def ccdf(self, x):
        '''
        Calculates the conditional cumulative distribution function.
        '''

    def ppcf(self, q):
        '''
        '''

    def fit(self, data):
        '''
        '''

    def rvs():
        '''
        '''
