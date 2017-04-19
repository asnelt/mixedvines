# -*- coding: utf-8 -*-
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
This module implements tests for the mixedvine module.
'''
from unittest import TestCase
from mixedvines.mixedvine import Marginal, MixedVine
import numpy as np
from numpy.testing import assert_allclose


class MarginalTestCase(TestCase):
    '''
    This class represents test cases for the Marginal class.
    '''
    def setUp(self):
        '''
        Saves the current random state for later recovery and sets the random
        seed to get reproducible results.
        '''
        # Save random state for later recovery
        self.random_state = np.random.get_state()
        # Set fixed random seed
        np.random.seed(0)

    def tearDown(self):
        '''
        Recovers the original random state.
        '''
        # Recover original random state
        np.random.set_state(self.random_state)

    def test_fit(self):
        '''
        Tests the fit method.
        '''
        samples = np.linspace(-2, 2, 3)
        # Normal distribution
        m = Marginal.fit(samples, True)
        # Comparison values
        r_logpdf = np.array([-2.15935316, -1.40935316, -2.15935316])
        p_logpdf = m.logpdf(samples)
        assert_allclose(p_logpdf, r_logpdf)


class MixedVineTestCase(TestCase):
    '''
    This class represents test cases for the MixedVine class.
    '''
    def setUp(self):
        '''
        Saves the current random state for later recovery and sets the random
        seed to get reproducible results.
        '''
        # Save random state for later recovery
        self.random_state = np.random.get_state()
        # Set fixed random seed
        np.random.seed(0)

    def tearDown(self):
        '''
        Recovers the original random state.
        '''
        # Recover original random state
        np.random.set_state(self.random_state)
