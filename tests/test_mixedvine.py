# Copyright (C) 2017-2019, 2021, 2022 Arno Onken
#
# This file is part of the mixedvines package.
#
# The mixedvines package is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# The mixedvines package is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""This module implements tests for the mixedvine module."""
import pytest
import numpy as np
from numpy.testing import assert_approx_equal, assert_allclose
from scipy.stats import norm, gamma, poisson
from mixedvines.copula import GaussianCopula, ClaytonCopula, FrankCopula
from mixedvines.mixedvine import MixedVine


@pytest.fixture(name="example_vine")
def fixture_example_vine():
    """Constructs an example mixed vine.

    Returns
    -------
    vine : MixedVine
        An example mixed vine.
    """
    dim = 3  # Dimension
    vine = MixedVine(dim)
    # Specify marginals
    vine.set_marginal(0, norm(0, 1))
    vine.set_marginal(1, poisson(5))
    vine.set_marginal(2, gamma(2, 0, 4))
    # Specify pair-copulas
    vine.set_copula(1, 0, GaussianCopula(0.5))
    vine.set_copula(1, 1, FrankCopula(4))
    vine.set_copula(2, 0, ClaytonCopula(5))
    return vine


def test_pdf(example_vine):
    """Tests the probability density function."""
    # Calculate probability density function on lattice
    bnds = np.empty((3), dtype=object)
    bnds[0] = [-1, 1]
    bnds[1] = [0, 2]
    bnds[2] = [0.5, 2]
    x0g, x1g, x2g = np.mgrid[bnds[0][0]:bnds[0][1],
                             bnds[1][0]:bnds[1][1],
                             bnds[2][0]:bnds[2][1]]
    points = np.array([x0g.ravel(), x1g.ravel(), x2g.ravel()]).T
    r_logpdf = np.array([-6.313469, -17.406428, -4.375992, -6.226508,
                         -8.836115, -20.430739, -5.107053, -6.687987])
    p_logpdf = example_vine.logpdf(points)
    assert_allclose(p_logpdf, r_logpdf)
    r_pdf = np.array([1.811738e-03, 2.757302e-08, 1.257566e-02,
                      1.976342e-03, 1.453865e-04, 1.339808e-09,
                      6.053895e-03, 1.245788e-03])
    p_pdf = example_vine.pdf(points)
    assert_allclose(p_pdf, r_pdf, rtol=1e-5)


def test_fit(example_vine):
    """Tests the fit to samples."""
    # Generate random variates
    size = 100
    random_state = np.random.RandomState(0)
    samples = example_vine.rvs(size=size, random_state=random_state)
    is_continuous = example_vine.is_continuous()
    # Fit mixed vine to samples
    vine_est = MixedVine.fit(samples, is_continuous)
    assert_approx_equal(vine_est.root.copulas[0].theta, 0.52951,
                        significant=5)
    assert_approx_equal(vine_est.root.input_layer.copulas[0].theta,
                        11.88942, significant=5)
    assert_approx_equal(vine_est.root.input_layer.copulas[1].theta,
                        4.56877, significant=5)


def test_entropy(example_vine):
    """Tests the entropy estimate."""
    random_state = np.random.RandomState(0)
    ent, sem = example_vine.entropy(sem_tol=1e-2, random_state=random_state)
    assert_approx_equal(ent, 7.83, significant=3)
    assert_approx_equal(sem, 0.00999, significant=3)
