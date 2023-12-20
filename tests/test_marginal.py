# Copyright (C) 2017-2019, 2021-2023 Arno Onken
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
"""This module implements tests for the marginal module."""
import numpy as np
from numpy.testing import assert_allclose
from mixedvines.marginal import Marginal


def test_marginal_fit():
    """Tests the fit method."""
    samples = np.linspace(-2, 2, 3)
    # Normal distribution
    marginal = Marginal.fit(samples, True)
    # Comparison values
    r_logpdf = np.array([-2.15935316, -1.40935316, -2.15935316])
    p_logpdf = marginal.logpdf(samples)
    assert_allclose(p_logpdf, r_logpdf)
