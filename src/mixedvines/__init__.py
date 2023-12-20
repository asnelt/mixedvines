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
"""
mixedvines
==========

Provides canonical vine copula trees with mixed continuous and discrete
marginals.  The main class is `MixedVine` implementing a copula vine model
with mixed marginals.

Modules
-------
mixedvine
    Copula vine model with mixed marginals.
copula
    Bivariate copula distributions.
marginal
    Univariate marginal distributions.

"""
from . import marginal
from . import copula
from . import mixedvine
from .mixedvine import MixedVine


__all__ = ['marginal', 'copula', 'mixedvine', 'MixedVine']
