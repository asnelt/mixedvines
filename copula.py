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
import numpy as np

class Copula(object):
    '''
    This class represents a copula.
    '''
    def __init__(self, family, parameter):
        '''
        Constructs a copula of a given family.
        '''
        self.family = family
        self.parameter = parameter

    def logpdf(self, x):
        '''
        '''

    def pdf(self, x):
        '''
        '''

    def logcdf(self, x):
        '''
        '''

    def cdf(self, x):
        '''
        '''
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
