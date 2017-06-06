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
from scipy.stats import norm, gamma, poisson
from mixedvines.copula import GaussianCopula, ClaytonCopula, FrankCopula
from mixedvines.mixedvine import MixedVine
import numpy as np
from numpy.testing import assert_approx_equal, assert_allclose


class MixedVineTestCase(TestCase):
    '''
    This class represents test cases for the MixedVine class.
    '''
    def setUp(self):
        '''
        Saves the current random state for later recovery, sets the random seed
        to get reproducible results and manually constructs a mixed vine.
        '''
        # Save random state for later recovery
        self.random_state = np.random.get_state()
        # Set fixed random seed
        np.random.seed(0)
        # Manually construct mixed vine
        self.dim = 3  # Dimension
        self.vine = MixedVine(self.dim)
        # Specify marginals
        self.vine.set_marginal(0, norm(0, 1))
        self.vine.set_marginal(1, poisson(5))
        self.vine.set_marginal(2, gamma(2, 0, 4))
        # Specify pair copulas
        self.vine.set_copula(1, 0, GaussianCopula(0.5))
        self.vine.set_copula(1, 1, FrankCopula(4))
        self.vine.set_copula(2, 0, ClaytonCopula(5))

    def tearDown(self):
        '''
        Recovers the original random state.
        '''
        # Recover original random state
        np.random.set_state(self.random_state)

    def test_pdf(self):
        '''
        Tests the probability density function.
        '''
        # Calculate probability density function on lattice
        bnds = np.empty((3), dtype=object)
        bnds[0] = [-1, 1]
        bnds[1] = [0, 2]
        bnds[2] = [0.5, 2]
        (x0g, x1g, x2g) = np.mgrid[bnds[0][0]:bnds[0][1],
                                   bnds[1][0]:bnds[1][1],
                                   bnds[2][0]:bnds[2][1]]
        points = np.array([x0g.ravel(), x1g.ravel(), x2g.ravel()]).T
        r_logpdf = np.array([-6.313469, -17.406428, -4.375992, -6.226508,
                             -8.836115, -20.430739, -5.107053, -6.687987])
        p_logpdf = self.vine.logpdf(points)
        assert_allclose(p_logpdf, r_logpdf)
        r_pdf = np.array([1.811738e-03, 2.757302e-08, 1.257566e-02,
                          1.976342e-03, 1.453865e-04, 1.339808e-09,
                          6.053895e-03, 1.245788e-03])
        p_pdf = self.vine.pdf(points)
        assert_allclose(p_pdf, r_pdf, rtol=1e-5)

    def test_fit(self):
        '''
        Tests the fit to samples.
        '''
        # Generate random variates
        size = 100
        samples = self.vine.rvs(size)
        # Fit mixed vine to samples
        is_continuous = np.full((self.dim), True, dtype=bool)
        is_continuous[1] = False
        vine_est = MixedVine.fit(samples, is_continuous)
        assert_approx_equal(vine_est.root.copulas[0].theta, 0.77490,
                            significant=5)
        assert_approx_equal(vine_est.root.input_layer.copulas[0].theta,
                            4.01646, significant=5)
        assert_approx_equal(vine_est.root.input_layer.copulas[1].theta,
                            4.56877, significant=5)

    def test_entropy(self):
        '''
        Tests the entropy estimate.
        '''
        (ent, sem) = self.vine.entropy(sem_tol=1e-2)
        assert_approx_equal(ent, 7.83, significant=3)
        assert_approx_equal(sem, 0.00999, significant=3)
