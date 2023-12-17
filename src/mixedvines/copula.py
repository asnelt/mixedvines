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
"""This module implements bivariate copula distributions.

Classes
-------
Copula
    Abstract class representing a copula.
IndependenceCopula
    Independence copula.
GaussianCopula
    Copula from the Gaussian family.
ClaytonCopula
    Copula from the Clayton family.
FrankCopula
    Copula from the Frank family.

"""
import abc
from scipy.optimize import minimize
from scipy.stats import norm, uniform, multivariate_normal
import numpy as np
from ._utils import select_best_dist


class Copula(abc.ABC):
    """This abstract class represents a copula.

    Parameters
    ----------
    theta : array_like or float, optional
        Parameter or parameter array of the copula.  The number of elements
        depends on the copula family.  (Default: `None`)
    rotation : string, optional
        Clockwise rotation of the copula.  Can be one of the elements of
        `Copula.rotation_options` or `None`.  (Default: `None`)

    Attributes
    ----------
    theta : array_like or float
        Parameter or parameter array of the copula.
    rotation : string
        Clockwise rotation of the copula.
    """

    rotation_options = ['0°', '90°', '180°', '270°']

    def __init__(self, theta=None, rotation=None):
        self.__check_theta(theta)
        self.__check_rotation(rotation)
        self.theta = theta
        self.rotation = rotation

    @classmethod
    def __check_theta(cls, theta):
        """Checks the `theta` parameter.

        Parameters
        ----------
        theta : array_like or float
            Parameter or parameter array of the copula.  The number of
            elements depends on the copula family.

        Raises
        ------
        ValueError
            If `theta` has the wrong size or is out of bounds.
        """
        bnds = cls.theta_bounds()
        if bnds:
            theta = np.asarray(theta)
            if theta.size != len(bnds):
                raise ValueError("the number of elements of 'theta' does not"
                                 " match the number of family parameters")
            if theta.size == 1:
                if theta < bnds[0][0] or theta > bnds[0][1]:
                    raise ValueError("parameter 'theta' out of bounds")
            else:
                for i, bnd in enumerate(bnds):
                    if theta[i] < bnd[0] or theta[i] > bnd[1]:
                        raise ValueError(f"parameter 'theta[{i}]' out of"
                                         " bounds")
        elif theta is not None:
            raise ValueError("for this copula family, 'theta' must be 'None'")

    @classmethod
    def __check_rotation(cls, rotation):
        """Checks the `rotation` parameter.

        Parameters
        ----------
        rotation : string
            Rotation of the copula.  Can be one of the elements of
            `Copula.rotation_options` or `None`.

        Raises
        ------
        ValueError
            If the `rotation` parameter is unsuitable.
        """
        if rotation is not None and rotation not in cls.rotation_options:
            raise ValueError("rotation '" + rotation + "' not supported")

    @staticmethod
    def __crop_input(samples):
        """Crops the input to the unit hypercube.

        The input is changed and a reference to the input is returned.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        samples : array_like
            n-by-2 matrix of cropped samples where n is the number of
            samples.
        """
        samples[samples < 0] = 0
        samples[samples > 1] = 1
        return samples

    def __rotate_input(self, samples):
        """Preprocesses the input to account for the copula rotation.

        The input is changed and a reference to the input is returned.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        samples : array_like
            n-by-2 matrix of rotated samples where n is the number of
            samples.
        """
        match self.rotation:
            case '90°':
                samples[:, 1] = 1 - samples[:, 1]
            case '180°':
                samples[:, :] = 1 - samples[:, :]
            case '270°':
                samples[:, 0] = 1 - samples[:, 0]
        return samples

    @abc.abstractmethod
    def _logpdf(self, samples):
        """Calculates the log of the probability density function.

        The samples can be assumed to lie within the unit hypercube.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        ndarray
            Log of the probability density function evaluated at `samples`.
        """

    def logpdf(self, samples):
        """Calculates the log of the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Log of the probability density function evaluated at `samples`.
        """
        samples = np.copy(np.asarray(samples))
        samples = self.__rotate_input(samples)
        inner = np.all(np.bitwise_and(samples > 0.0, samples < 1.0), axis=1)
        outer = np.invert(inner)
        vals = np.zeros(samples.shape[0])
        vals[inner] = self._logpdf(samples[inner, :])
        # Assign zero mass to border
        vals[outer] = -np.inf
        return vals

    def pdf(self, samples):
        """Calculates the probability density function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        ndarray
            Probability density function evaluated at `samples`.
        """
        return np.exp(self.logpdf(samples))

    @abc.abstractmethod
    def _logcdf(self, samples):
        """Calculates the log of the cumulative distribution function.

        The samples can be assumed to lie within the unit hypercube.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        ndarray
            Log of the cumulative distribution function evaluated at
            `samples`.
        """

    def logcdf(self, samples):
        """Calculates the log of the cumulative distribution function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        vals : ndarray
            Log of the cumulative distribution function evaluated at
            `samples`.
        """
        samples = np.copy(np.asarray(samples))
        samples = self.__crop_input(samples)
        samples = self.__rotate_input(samples)
        vals = self._logcdf(samples)
        # Transform according to rotation, but take `__rotate_input` into
        # account.
        match self.rotation:
            case '90°':
                with np.errstate(divide='ignore'):
                    vals = np.log(np.maximum(0, samples[:, 0] - np.exp(vals)))
            case '180°':
                with np.errstate(divide='ignore'):
                    vals = np.log(np.maximum(0,
                                             (1.0 - samples[:, 0])
                                             + (1.0 - samples[:, 1])
                                             - 1.0 + np.exp(vals)))
            case '270°':
                with np.errstate(divide='ignore'):
                    vals = np.log(np.maximum(0, samples[:, 1] - np.exp(vals)))
        return vals

    def cdf(self, samples):
        """Calculates the cumulative distribution function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        ndarray
            Cumulative distribution function evaluated at `samples`.
        """
        return np.exp(self.logcdf(samples))

    def __axis_wrapper(self, fun, samples, axis):
        """Calls function `fun` with `samples` as argument.

        Eventually changes rotation and samples such that `axis == 0`
        corresponds to `axis == 1`.

        Parameters
        ----------
        fun : function
            Function to be called with `samples` as argument.
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        axis : int
            The axis to condition the cumulative distribution function on.

        Returns
        -------
        vals : array_like
            Function values evaluated at `samples` but taking `axis` into
            account.

        Raises
        ------
        ValueError
            If `axis` is not 0 and not 1.
        """
        samples = np.copy(np.asarray(samples))
        samples = self.__crop_input(samples)
        rotation = self.rotation
        try:
            # Temporarily change rotation according to axis
            if axis == 0:
                if self.rotation == '90°':
                    self.rotation = '270°'
                elif self.rotation == '270°':
                    self.rotation = '90°'
                samples = samples[:, [1, 0]]
            elif axis != 1:
                raise ValueError("'axis' must be in [0, 1]")
            samples = self.__rotate_input(samples)
            vals = fun(samples)
            if self.rotation in ('180°', '270°'):
                vals = 1.0 - vals
        finally:
            # Recover original rotation
            self.rotation = rotation
        return vals

    @abc.abstractmethod
    def _ccdf(self, samples):
        """Calculates the conditional cumulative distribution function.

        The cumulative distribution function is conditioned on axis 1.  The
        samples can be assumed to lie within the unit hypercube.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        ndarray
            Conditional cumulative distribution function evaluated at
            `samples`.
        """

    def ccdf(self, samples, axis=1):
        """Calculates the conditional cumulative distribution function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        axis : int, optional
            The axis to condition the cumulative distribution function on.
            (Default: 1)

        Returns
        -------
        ndarray
            Conditional cumulative distribution function evaluated at
            `samples`.
        """
        return self.__axis_wrapper(self._ccdf, samples, axis)

    @abc.abstractmethod
    def _ppcf(self, samples):
        """Calculates the inverse of the conditional CDF.

        Calculates the inverse of the copula conditional cumulative
        distribution function conditioned on axis 1.  The samples can be
        assumed to lie within the unit hypercube.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        ndarray
            Inverse of the conditional cumulative distribution function
            evaluated at `samples`.
        """

    def ppcf(self, samples, axis=1):
        """Calculates the inverse of the conditional CDF.

        Calculates the inverse of the copula conditional cumulative
        distribution function.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        axis : int, optional
            The axis to condition the cumulative distribution function on.
            (Default: 1)

        Returns
        -------
        ndarray
            Inverse of the conditional cumulative distribution function
            evaluated at `samples`.
        """
        return self.__axis_wrapper(self._ppcf, samples, axis)

    def rvs(self, size=1, random_state=None):
        """Generates random variates from the copula.

        Parameters
        ----------
        size : int, optional
            The number of samples to generate.  (Default: 1)
        random_state : {None, int, `numpy.random.Generator`,
                        `numpy.random.RandomState`}, optional

            The random state to use for random variate generation.  `None`
            corresponds to the `RandomState` singleton.  For an `int`, a
            new `RandomState` is generated and seeded.  For a
            `RandomState` or `Generator`, the object is used.
            (Default: `None`)

        Returns
        -------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        """
        samples = np.stack((uniform.rvs(size=size, random_state=random_state),
                            uniform.rvs(size=size, random_state=random_state)),
                           axis=1)
        samples[:, 0] = self.ppcf(samples)
        return samples

    def estimate_theta(self, samples):
        """Estimates the theta parameters from the given samples.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.
        """
        if self.theta is not None:
            bnds = self.theta_bounds()

            def cost(theta):
                """Calculates the cost of a given `theta` parameter."""
                self.theta = theta
                vals = self.logpdf(samples)
                # For optimization, filter out infinity values
                return -np.sum(vals[np.isfinite(vals)])

            result = minimize(cost, self.theta, method='TNC', bounds=bnds)
            self.theta = result.x
            if len(result.x) == 1:
                self.theta = result.x[0]
            else:
                self.theta = result.x

    @classmethod
    def fit(cls, samples):
        """Fits the parameters of the copula to the given samples.

        Parameters
        ----------
        samples : array_like
            n-by-2 matrix of samples where n is the number of samples.

        Returns
        -------
        copula : Copula
            The copula fitted to `samples`.
        """
        # Find best fitting family
        copulas = [family.fit(samples) for family in cls.__subclasses__()]
        param_counts = [0 if copula.theta is None
                        else 1 if np.isscalar(copula.theta)
                        else len(copula.theta)
                        for copula in copulas]
        # Select best copula
        best_copula = select_best_dist(samples, copulas, param_counts)
        return best_copula

    @staticmethod
    @abc.abstractmethod
    def theta_bounds():
        """Bounds for `theta` parameters.

        Returns
        -------
        array_like
            List of 2-tuples where the first tuple element represents the
            lower bound and the second element represents the upper bound.
        """


class IndependenceCopula(Copula):
    """This class represents the independence copula."""

    def _logpdf(self, samples):
        vals = np.zeros(samples.shape[0])
        return vals

    def _logcdf(self, samples):
        with np.errstate(divide='ignore'):
            vals = np.sum(np.log(samples), axis=1)
        return vals

    def _ccdf(self, samples):
        vals = samples[:, 0]
        return vals

    def _ppcf(self, samples):
        vals = samples[:, 0]
        return vals

    @classmethod
    def fit(cls, samples):
        copula = cls()
        return copula

    @staticmethod
    def theta_bounds():
        bnds = []
        return bnds


class GaussianCopula(Copula):
    """This class represents a copula from the Gaussian family."""

    def _logpdf(self, samples):
        if self.theta >= 1.0:
            vals = np.zeros(samples.shape[0])
            vals[samples[:, 0] == samples[:, 1]] = np.inf
        elif self.theta <= -1.0:
            vals = np.zeros(samples.shape[0])
            vals[samples[:, 0] == 1 - samples[:, 1]] = np.inf
        else:
            nrvs = norm.ppf(samples)
            vals = 2 * self.theta * nrvs[:, 0] * nrvs[:, 1] - self.theta**2 \
                * (nrvs[:, 0]**2 + nrvs[:, 1]**2)
            vals /= 2 * (1 - self.theta**2)
            vals -= np.log(1 - self.theta**2) / 2
        return vals

    def _logcdf(self, samples):
        upper = norm.ppf(samples)
        cov = [[1.0, self.theta], [self.theta, 1.0]]
        vals = multivariate_normal.logcdf(upper, None, cov)
        vals[np.any(samples == 0.0, axis=1)] = -np.inf
        vals[samples[:, 0] == 1.0] = np.log(samples[samples[:, 0] == 1.0, 1])
        vals[samples[:, 1] == 1.0] = np.log(samples[samples[:, 1] == 1.0, 0])
        return vals

    def _ccdf(self, samples):
        vals = np.zeros(samples.shape[0])
        # Avoid subtraction of infinities
        neqz = np.bitwise_and(np.any(samples > 0.0, axis=1),
                              np.any(samples < 1.0, axis=1))
        nrvs = norm.ppf(samples[neqz, :])
        vals[neqz] = norm.cdf((nrvs[:, 0] - self.theta * nrvs[:, 1])
                              / np.sqrt(1 - self.theta**2))
        vals[np.invert(neqz)] = norm.cdf(0.0)
        return vals

    def _ppcf(self, samples):
        nrvs = norm.ppf(samples)
        vals = norm.cdf(nrvs[:, 0] * np.sqrt(1 - self.theta**2)
                        + self.theta * nrvs[:, 1])
        return vals

    @classmethod
    def fit(cls, samples):
        initial_point = 0.0
        copula = cls(theta=initial_point)
        copula.estimate_theta(samples)
        return copula

    @staticmethod
    def theta_bounds():
        bnds = [(-1.0 + 1e-3, 1.0 - 1e-3)]
        return bnds


class ClaytonCopula(Copula):
    """This class represents a copula from the Clayton family."""

    def _logpdf(self, samples):
        if self.theta == 0:
            vals = np.zeros(samples.shape[0])
        else:
            vals = np.log1p(self.theta) + (-1 - self.theta) \
                   * (np.log(samples[:, 0]) + np.log(samples[:, 1])) \
                   + (-1 / self.theta - 2) \
                   * np.log(samples[:, 0]**(-self.theta)
                            + samples[:, 1]**(-self.theta) - 1)
        return vals

    def _logcdf(self, samples):
        if self.theta == 0:
            vals = np.sum(np.log(samples), axis=1)
        else:
            with np.errstate(divide='ignore'):
                vals = (-1 / self.theta) \
                    * np.log(np.maximum(samples[:, 0]**(-self.theta)
                                        + samples[:, 1]**(-self.theta) - 1,
                                        0))
        return vals

    def _ccdf(self, samples):
        if self.theta == 0:
            vals = samples[:, 0]
        else:
            vals = np.zeros(samples.shape[0])
            gtz = np.all(samples > 0.0, axis=1)
            vals[gtz] = np.maximum(samples[gtz, 1]**(-1 - self.theta)
                                   * (samples[gtz, 0]**(-self.theta)
                                      + samples[gtz, 1]**(-self.theta) - 1)
                                   ** (-1 - 1 / self.theta), 0)
        return vals

    def _ppcf(self, samples):
        if self.theta == 0:
            vals = samples[:, 0]
        else:
            vals = np.zeros(samples.shape[0])
            gtz = np.all(samples > 0.0, axis=1)
            vals[gtz] = (1 - samples[gtz, 1]**(-self.theta)
                         + (samples[gtz, 0]
                            * (samples[gtz, 1]**(1 + self.theta)))
                         ** (-self.theta / (1 + self.theta))) \
                ** (-1 / self.theta)
        return vals

    @classmethod
    def fit(cls, samples):
        initial_point = 1.0
        # Optimize rotation as well
        copulas = [cls(theta=initial_point, rotation=rotation)
                   for rotation in cls.rotation_options]
        # Fit parameters
        for copula in copulas:
            copula.estimate_theta(samples)
        param_counts = [0 if copula.theta is None
                        else 1 if np.isscalar(copula.theta)
                        else len(copula.theta)
                        for copula in copulas]
        # Select best copula
        best_copula = select_best_dist(samples, copulas, param_counts)
        return best_copula

    @staticmethod
    def theta_bounds():
        bnds = [(1e-3, 20)]
        return bnds


class FrankCopula(Copula):
    """This class represents a copula from the Frank family."""

    def _logpdf(self, samples):
        if self.theta == 0:
            vals = np.zeros(samples.shape[0])
        else:
            vals = np.log(-self.theta * np.expm1(-self.theta)
                          * np.exp(-self.theta
                                   * (samples[:, 0] + samples[:, 1]))
                          / (np.expm1(-self.theta)
                             + np.expm1(-self.theta * samples[:, 0])
                             * np.expm1(-self.theta * samples[:, 1])) ** 2)
        return vals

    def _logcdf(self, samples):
        if self.theta == 0:
            vals = np.sum(np.log(samples), axis=1)
        else:
            with np.errstate(divide='ignore'):
                vals = np.log(-np.log1p(np.expm1(-self.theta * samples[:, 0])
                                        * np.expm1(-self.theta * samples[:, 1])
                                        / (np.expm1(-self.theta)))) \
                    - np.log(self.theta)
        return vals

    def _ccdf(self, samples):
        if self.theta == 0:
            vals = samples[:, 0]
        else:
            vals = np.exp(-self.theta * samples[:, 1]) \
                * np.expm1(-self.theta * samples[:, 0]) \
                / (np.expm1(-self.theta)
                   + np.expm1(-self.theta * samples[:, 0])
                   * np.expm1(-self.theta * samples[:, 1]))
        return vals

    def _ppcf(self, samples):
        if self.theta == 0:
            vals = samples[:, 0]
        else:
            vals = -np.log1p(samples[:, 0] * np.expm1(-self.theta)
                             / (np.exp(-self.theta * samples[:, 1])
                                - samples[:, 0] * np.expm1(-self.theta
                                                           * samples[:, 1]))) \
                / self.theta
        return vals

    @classmethod
    def fit(cls, samples):
        initial_point = 0.0
        copula = cls(theta=initial_point)
        copula.estimate_theta(samples)
        return copula

    @staticmethod
    def theta_bounds():
        bnds = [(-20, 20)]
        return bnds
