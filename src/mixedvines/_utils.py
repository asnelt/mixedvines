# Copyright (C) 2017-2026 Arno Onken
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
"""This module implements helper functions for the mixedvines package."""
import numpy as np


def select_best_dist(samples, dists, param_counts):
    """Selects the best distribution for given samples.

    Uses the Akaike information criterion to assess each distribution in
    `dists`.

    Parameters
    ----------
    samples : array_like
        n-by-d matrix of samples where n is the number of samples and d is
        the size of each sample.
    dists : array_like
        Distribution options.
    param_counts : array_like
        The number of parameters of each distribution.  Must have the same
        length as `dists`.

    Returns
    -------
    best_dist : Marginal or Copula
        The best distribution from `dists`.
    """
    logpdfs = np.array([dist.logpdf(samples) for dist in dists])
    # Indices that are informative for selection
    informative = np.any(np.isfinite(logpdfs), axis=0)
    with np.errstate(invalid='ignore'):
        # Log-likelihood based on informative samples
        logl = np.sum(logpdfs[:, informative], axis=1)
    aic = np.where(np.isfinite(logl), 2 * np.asarray(param_counts) - 2 * logl,
                   np.inf)
    best_dist = dists[np.argmin(aic)]
    return best_dist
