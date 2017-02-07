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
This module implements tests for the copula module.
'''
from unittest import TestCase
from mixedvines.copula import Copula
import numpy as np

class CopulaTestCase(TestCase):
    '''
    This class represents test cases for the Copula class.
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

    def test_logpdf(self):
        '''
        Tests the log of the probability density function.
        '''
        u = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

        # Independence copula
        c = Copula('ind')
        # Comparison values obtained from R
        r_logpdf = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        p_logpdf = c.logpdf(u)
        self.assertAlmostEqual(p_logpdf, r_logpdf)

        # Gaussian copula family
        c = Copula('gaussian', 0.5)
        # Comparison values obtained from R
        r_logpdf = np.array([-np.inf, 0.2165361255, 0.1438410362, \
                             0.2165361255, -np.inf])
        p_logpdf = c.logpdf(u)
        self.assertAlmostEqual(p_logpdf, r_logpdf)

        # Student t copula family
        c = Copula('student', [0.5, 10])
        # Comparison values obtained from R
        r_logpdf = np.array([-np.inf, 0.2540239468, 0.1937586741, \
                             0.2540239468, -np.inf])
        p_logpdf = c.logpdf(u)
        self.assertAlmostEqual(p_logpdf, r_logpdf)

        # Clayton copula family
        c = Copula('clayton', 5)
        # Comparison values obtained from R
        r_logpdf = np.array([-np.inf, 0.7858645247, 0.9946292379, \
                             0.6666753203, -np.inf])
        p_logpdf = c.logpdf(u)
        self.assertAlmostEqual(p_logpdf, r_logpdf)

    def test_pdf(self):
        '''
        Tests the probability density function.
        '''
        u = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

        # Independence copula
        c = Copula('ind')
        # Comparison values obtained from R
        r_pdf = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        p_pdf = c.pdf(u)
        self.assertAlmostEqual(p_pdf, r_pdf)

        # Gaussian copula family
        c = Copula('gaussian', 0.5)
        # Comparison values obtained from R
        r_logpdf = np.array([0.0, 1.2417679440, 1.1547005384, 1.2417679440,
                             0.0])
        p_logpdf = c.logpdf(u)
        self.assertAlmostEqual(p_pdf, r_pdf)

        # Student t copula family
        c = Copula('student', [0.5, 10])
        # Comparison values obtained from R
        r_logpdf = np.array([0.0, 1.2892026762, 1.2138033255, 1.2892026762, \
                             0.0])
        p_logpdf = c.logpdf(u)
        self.assertAlmostEqual(p_pdf, r_pdf)

        # Clayton copula family
        c = Copula('clayton', 5)
        # Comparison values obtained from R
        r_logpdf = np.array([0.0, 2.1943031503, 2.7037217178, 1.9477508961, \
                             0.0])
        p_logpdf = c.logpdf(u)
        self.assertAlmostEqual(p_pdf, r_pdf)
