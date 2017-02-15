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
from numpy.testing import assert_allclose


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
        assert_allclose(p_logpdf, r_logpdf)

        # Gaussian copula family
        c = Copula('gaussian', 0.5)
        # Comparison values obtained from R
        r_logpdf = np.array([-np.inf, 0.2165361255, 0.1438410362,
                             0.2165361255, -np.inf])
        p_logpdf = c.logpdf(u)
        assert_allclose(p_logpdf, r_logpdf)

        # Student t copula family
        c = Copula('student', [0.5, 10])
        # Comparison values obtained from R
        r_logpdf = np.array([-np.inf, 0.1491425169, 0.1937586741,
                             0.1491425169, -np.inf])
        p_logpdf = c.logpdf(u)
        assert_allclose(p_logpdf, r_logpdf)

        # Clayton copula family
        c = Copula('clayton', 5)
        # Comparison values obtained from R
        r_logpdf = np.array([-np.inf, 0.7858645247, 0.9946292379,
                             0.6666753203, -np.inf])
        p_logpdf = c.logpdf(u)
        assert_allclose(p_logpdf, r_logpdf)

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
        assert_allclose(p_pdf, r_pdf)

        # Gaussian copula family
        c = Copula('gaussian', 0.5)
        # Comparison values obtained from R
        r_pdf = np.array([0.0, 1.2417679440, 1.1547005384, 1.2417679440, 0.0])
        p_pdf = c.pdf(u)
        assert_allclose(p_pdf, r_pdf)

        # Student t copula family
        c = Copula('student', [0.5, 10])
        # Comparison values obtained from R
        r_pdf = np.array([0.0, 1.1608384165, 1.2138033255, 1.1608384165, 0.0])
        p_pdf = c.pdf(u)
        assert_allclose(p_pdf, r_pdf)

        # Clayton copula family
        c = Copula('clayton', 5)
        # Comparison values obtained from R
        r_pdf = np.array([0.0, 2.1943031503, 2.7037217178, 1.9477508961, 0.0])
        p_pdf = c.pdf(u)
        assert_allclose(p_pdf, r_pdf)

    def test_logcdf(self):
        '''
        Tests the log of the cumulative distribution function.
        '''
        u = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

        # Independence copula
        c = Copula('ind')
        # Comparison values obtained from R
        r_logcdf = np.array([-np.inf, -2.4361164856, -1.3862943611,
                             -0.7184649885, -0.2231435513])
        p_logcdf = c.logcdf(u)
        assert_allclose(p_logcdf, r_logcdf)

        # Gaussian copula family
        c = Copula('gaussian', 0.5)
        # Comparison values obtained from R
        r_logcdf = np.array([-np.inf, -1.8836553477, -1.0986122887,
                             -0.5941468105, -0.2231435513])
        p_logcdf = c.logcdf(u)
        assert_allclose(p_logcdf, r_logcdf)

        # Clayton copula family
        c = Copula('clayton', 5)
        # Comparison values obtained from R
        r_logcdf = np.array([-np.inf, -1.4202358053, -0.8286269453,
                             -0.4941703709, -0.2231435513])
        p_logcdf = c.logcdf(u)
        assert_allclose(p_logcdf, r_logcdf)

    def test_cdf(self):
        '''
        Tests the cumulative distribution function.
        '''
        u = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

        # Independence copula
        c = Copula('ind')
        # Comparison values obtained from R
        r_cdf = np.array([0.0, 0.0875, 0.25, 0.4875, 0.8])
        p_cdf = c.cdf(u)
        assert_allclose(p_cdf, r_cdf)

        # Gaussian copula family
        c = Copula('gaussian', 0.5)
        # Comparison values obtained from R
        r_cdf = np.array([0.0, 0.1520333540, 0.3333333333, 0.5520333540, 0.8])
        p_cdf = c.cdf(u)
        assert_allclose(p_cdf, r_cdf)

        # Clayton copula family
        c = Copula('clayton', 5)
        # Comparison values obtained from R
        r_cdf = np.array([0.0, 0.2416570262, 0.4366484171, 0.6100768349, 0.8])
        p_cdf = c.cdf(u)
        assert_allclose(p_cdf, r_cdf)

    def test_ccdf(self):
        '''
        Tests the conditional cumulative distribution function.
        '''
        u = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

        # Independence copula
        c = Copula('ind')
        # Comparison values obtained from R
        r_ccdf = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        p_ccdf = c.ccdf(u)
        assert_allclose(p_ccdf, r_ccdf)
        # Test other axis
        r_ccdf = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        p_ccdf = c.ccdf(u, axis=0)
        assert_allclose(p_ccdf, r_ccdf)

        # Gaussian copula family
        c = Copula('gaussian', 0.5)
        # Comparison values obtained from R
        r_ccdf = np.array([0.0, 0.2889793807, 0.5, 0.7110206193, 1.0])
        p_ccdf = c.ccdf(u)
        assert_allclose(p_ccdf, r_ccdf)
        # Test other axis
        r_ccdf = np.array([1.0, 0.4778649221, 0.5, 0.5221350779, 0.0])
        p_ccdf = c.ccdf(u, axis=0)
        assert_allclose(p_ccdf, r_ccdf)

        # Student t copula family
        c = Copula('student', [0.5, 10])
        # Comparison values obtained from R
        r_ccdf = np.array([0.0, 0.2794817821, 0.5, 0.7205182179, 1.0])
        p_ccdf = c.ccdf(u)
        assert_allclose(p_ccdf, r_ccdf)
        # Test other axis
        r_ccdf = np.array([0.9590678844, 0.4784831978, 0.5, 0.5215168022,
                           0.0409321156])
        p_ccdf = c.ccdf(u, axis=0)
        assert_allclose(p_ccdf, r_ccdf)

        # Clayton copula family
        c = Copula('clayton', 5)
        # Comparison values obtained from R
        r_ccdf = np.array([0.0, 0.1083398661, 0.4435793443, 0.6836393756,
                           1.0])
        p_ccdf = c.ccdf(u)
        assert_allclose(p_ccdf, r_ccdf)
        # Test other axis
        r_ccdf = np.array([0.0, 0.815748922, 0.4435793443, 0.2896940854,
                           0.262144])
        p_ccdf = c.ccdf(u, axis=0)
        assert_allclose(p_ccdf, r_ccdf)
