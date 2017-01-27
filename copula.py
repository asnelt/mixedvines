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
from scipy.stats import norm
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
            val = 0 # TODO
        elif self.family == 'clayton':
            if self.theta == 0:
                val = 0.0
            else:
                val = np.log(1 + self.theta) \
                        + (-1 - self.theta) * (np.log(u[:, 0]) + np.log(u[:, 1])) \
                        + (-1 / self.theta-2) * np.log(u[:, 0]**(-self.theta) + u[:, 1]**(-self.theta) - 1)
        return val

    def pdf(self, u):
        '''
        '''
        return np.exp(self.logpdf(u))

    def logcdf(self, u):
        '''
        '''

    def cdf(self, u):
        '''
        '''
        return np.exp(self.logcdf(u))

    def ccdf(self, x):
        '''
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
