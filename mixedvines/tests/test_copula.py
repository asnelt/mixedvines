# -*- coding: utf-8 -*-
# Copyright (C) 2017-2019, 2021 Arno Onken
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
"""This module implements tests for the copula module.i"""
import numpy as np
from numpy.testing import assert_allclose
from mixedvines.copula import IndependenceCopula, GaussianCopula, \
        ClaytonCopula, FrankCopula


def test_logpdf():
    """Tests the log of the probability density function."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Independence copula
    independence_copula = IndependenceCopula()
    # Comparison values
    r_logpdf = np.array([-np.inf, 0.0, 0.0, 0.0, -np.inf])
    p_logpdf = independence_copula.logpdf(samples)
    assert_allclose(p_logpdf, r_logpdf)

    # Gaussian copula family
    gaussian_copula = GaussianCopula(0.5)
    # Comparison values
    r_logpdf = np.array([-np.inf, 0.2165361255, 0.1438410362,
                         0.2165361255, -np.inf])
    p_logpdf = gaussian_copula.logpdf(samples)
    assert_allclose(p_logpdf, r_logpdf)

    # Clayton copula family
    clayton_copula = ClaytonCopula(5)
    # Comparison values
    r_logpdf = np.array([-np.inf, 0.7858645247, 0.9946292379,
                         0.6666753203, -np.inf])
    p_logpdf = clayton_copula.logpdf(samples)
    assert_allclose(p_logpdf, r_logpdf)

    # Frank copula family
    frank_copula = FrankCopula(5)
    # Comparison values
    r_logpdf = np.array([-np.inf, 0.4165775202, 0.3876837693, 0.4165775202,
                         -np.inf])
    p_logpdf = frank_copula.logpdf(samples)
    assert_allclose(p_logpdf, r_logpdf)


def test_pdf():
    """Tests the probability density function."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Independence copula
    independence_copula = IndependenceCopula()
    # Comparison values
    r_pdf = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    p_pdf = independence_copula.pdf(samples)
    assert_allclose(p_pdf, r_pdf)

    # Gaussian copula family
    gaussian_copula = GaussianCopula(0.5)
    # Comparison values
    r_pdf = np.array([0.0, 1.2417679440, 1.1547005384, 1.2417679440, 0.0])
    p_pdf = gaussian_copula.pdf(samples)
    assert_allclose(p_pdf, r_pdf)

    # Clayton copula family
    clayton_copula = ClaytonCopula(5)
    # Comparison values
    r_pdf = np.array([0.0, 2.1943031503, 2.7037217178, 1.9477508961, 0.0])
    p_pdf = clayton_copula.pdf(samples)
    assert_allclose(p_pdf, r_pdf)

    # Frank copula family
    frank_copula = FrankCopula(5)
    # Comparison values
    r_pdf = np.array([0.0, 1.5167615765, 1.4735637246, 1.5167615765, 0.0])
    p_pdf = frank_copula.pdf(samples)
    assert_allclose(p_pdf, r_pdf)


def test_logcdf():
    """Tests the log of the cumulative distribution function."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Independence copula
    independence_copula = IndependenceCopula()
    # Comparison values
    r_logcdf = np.array([-np.inf, -2.4361164856, -1.3862943611,
                         -0.7184649885, -0.2231435513])
    p_logcdf = independence_copula.logcdf(samples)
    assert_allclose(p_logcdf, r_logcdf)

    # Gaussian copula family
    gaussian_copula = GaussianCopula(0.5)
    # Comparison values
    r_logcdf = np.array([-np.inf, -1.8836553477, -1.0986122887,
                         -0.5941468105, -0.2231435513])
    p_logcdf = gaussian_copula.logcdf(samples)
    assert_allclose(p_logcdf, r_logcdf)

    # Clayton copula family
    clayton_copula = ClaytonCopula(5)
    # Comparison values
    r_logcdf = np.array([-np.inf, -1.4202358053, -0.8286269453,
                         -0.4941703709, -0.2231435513])
    p_logcdf = clayton_copula.logcdf(samples)
    assert_allclose(p_logcdf, r_logcdf)

    # Frank copula family
    frank_copula = FrankCopula(5)
    # Comparison values
    r_logcdf = np.array([-np.inf, -1.7145879734, -0.9751162414,
                         -0.5446618572, -0.2231435513])
    p_logcdf = frank_copula.logcdf(samples)
    assert_allclose(p_logcdf, r_logcdf)


def test_cdf():
    """Tests the cumulative distribution function."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Independence copula
    independence_copula = IndependenceCopula()
    # Comparison values
    r_cdf = np.array([0.0, 0.0875, 0.25, 0.4875, 0.8])
    p_cdf = independence_copula.cdf(samples)
    assert_allclose(p_cdf, r_cdf)

    # Gaussian copula family
    gaussian_copula = GaussianCopula(0.5)
    # Comparison values
    r_cdf = np.array([0.0, 0.1520333540, 0.3333333333, 0.5520333540, 0.8])
    p_cdf = gaussian_copula.cdf(samples)
    assert_allclose(p_cdf, r_cdf)

    # Clayton copula family
    clayton_copula = ClaytonCopula(5)
    # Comparison values
    r_cdf = np.array([0.0, 0.2416570262, 0.4366484171, 0.6100768349, 0.8])
    p_cdf = clayton_copula.cdf(samples)
    assert_allclose(p_cdf, r_cdf)

    # Frank copula family
    frank_copula = FrankCopula(5)
    # Comparison values
    r_cdf = np.array([0.0, 0.1800378858, 0.3771485107, 0.5800378858, 0.8])
    p_cdf = frank_copula.cdf(samples)
    assert_allclose(p_cdf, r_cdf)


def test_ccdf():
    """Tests the conditional cumulative distribution function."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Independence copula
    independence_copula = IndependenceCopula()
    # Comparison values
    r_ccdf = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    p_ccdf = independence_copula.ccdf(samples)
    assert_allclose(p_ccdf, r_ccdf)
    # Test other axis
    r_ccdf = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    p_ccdf = independence_copula.ccdf(samples, axis=0)
    assert_allclose(p_ccdf, r_ccdf)

    # Gaussian copula family
    gaussian_copula = GaussianCopula(0.5)
    # Comparison values
    r_ccdf = np.array([0.0, 0.2889793807, 0.5, 0.7110206193, 1.0])
    p_ccdf = gaussian_copula.ccdf(samples)
    assert_allclose(p_ccdf, r_ccdf)
    # Test other axis
    r_ccdf = np.array([1.0, 0.4778649221, 0.5, 0.5221350779, 0.0])
    p_ccdf = gaussian_copula.ccdf(samples, axis=0)
    assert_allclose(p_ccdf, r_ccdf)

    # Clayton copula family
    clayton_copula = ClaytonCopula(5)
    # Comparison values
    r_ccdf = np.array([0.0, 0.1083398661, 0.4435793443, 0.6836393756,
                       1.0])
    p_ccdf = clayton_copula.ccdf(samples)
    assert_allclose(p_ccdf, r_ccdf)
    # Test other axis
    r_ccdf = np.array([0.0, 0.815748922, 0.4435793443, 0.2896940854,
                       0.262144])
    p_ccdf = clayton_copula.ccdf(samples, axis=0)
    assert_allclose(p_ccdf, r_ccdf)

    # Frank copula family
    frank_copula = FrankCopula(5)
    # Comparison values
    r_ccdf = np.array([0.0, 0.3070854, 0.5, 0.6929146, 1.0])
    p_ccdf = frank_copula.ccdf(samples)
    assert_allclose(p_ccdf, r_ccdf)
    # Test other axis
    r_ccdf = np.array([0.63640865, 0.58629237, 0.5, 0.41370763,
                       0.36359135])
    p_ccdf = frank_copula.ccdf(samples, axis=0)
    assert_allclose(p_ccdf, r_ccdf)


def test_ppcf():
    """Tests the conditional cumulative distribution function."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Independence copula
    independence_copula = IndependenceCopula()
    # Comparison values
    r_ppcf = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    p_ppcf = independence_copula.ppcf(samples)
    assert_allclose(p_ppcf, r_ppcf)
    # Test other axis
    r_ppcf = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    p_ppcf = independence_copula.ppcf(samples, axis=0)
    assert_allclose(p_ppcf, r_ppcf)

    # Gaussian copula family
    gaussian_copula = GaussianCopula(0.5)
    # Comparison values
    r_ppcf = np.array([0.0, 0.218642669, 0.5, 0.781357331, 1.0])
    p_ppcf = gaussian_copula.ppcf(samples)
    assert_allclose(p_ppcf, r_ppcf)
    # Test other axis
    r_ppcf = np.array([0.0, 0.2511286797, 0.5, 0.7488713203, 1.0])
    p_ppcf = gaussian_copula.ppcf(samples, axis=0)
    assert_allclose(p_ppcf, r_ppcf)

    # Clayton copula family
    clayton_copula = ClaytonCopula(5)
    # Comparison values
    r_ppcf = np.array([0.0, 0.2994846602, 0.5211309028, 0.7859307932, 1.0])
    p_ppcf = clayton_copula.ppcf(samples)
    assert_allclose(p_ppcf, r_ppcf)
    # Test other axis
    r_ppcf = np.array([0.0, 0.2337467913, 0.5211309028, 0.8127416749,
                       0.9634924840])
    p_ppcf = clayton_copula.ppcf(samples, axis=0)
    assert_allclose(p_ppcf, r_ppcf)

    # Frank copula family
    frank_copula = FrankCopula(5)
    # Comparison values
    r_ppcf = np.array([0.0, 0.21162507, 0.5, 0.78837493, 1.0])
    p_ppcf = frank_copula.ppcf(samples)
    assert_allclose(p_ppcf, r_ppcf)
    # Test other axis
    r_ppcf = np.array([0.0442921, 0.20900068, 0.5, 0.79099932, 0.9557079])
    p_ppcf = frank_copula.ppcf(samples, axis=0)
    assert_allclose(p_ppcf, r_ppcf)


def test_rotation_90_deg():
    """Tests the 90° copula rotation."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Clayton copula family rotated 90°
    copula = ClaytonCopula(5, rotation='90°')
    # Comparison values
    r_logpdf = np.array([-np.inf, -2.571322188, 0.9946292379,
                         -1.7680858282, -np.inf])
    r_pdf = np.array([0.0, 0.0764344179, 2.7037217178, 0.1706593477, 0.0])
    r_logcdf = np.array([-np.inf, -7.9010702481, -2.7590553856,
                         -0.9133704691, -0.2231435513])
    r_cdf = np.array([0.0, 0.000370347, 0.0633515829, 0.4011698124, 0.8])
    r_ccdf = np.array([0.0, 0.003208462, 0.4435793443, 0.9801128956, 1.0])
    r_ppcf = np.array([0.0, 0.5506995734, 0.5211309028, 0.452724971, 1.0])
    p_logpdf = copula.logpdf(samples)
    p_pdf = copula.pdf(samples)
    p_logcdf = copula.logcdf(samples)
    p_cdf = copula.cdf(samples)
    p_ccdf = copula.ccdf(samples)
    p_ppcf = copula.ppcf(samples)
    assert_allclose(p_logpdf, r_logpdf)
    assert_allclose(p_pdf, r_pdf)
    assert_allclose(p_logcdf, r_logcdf)
    assert_allclose(p_cdf, r_cdf)
    assert_allclose(p_ccdf, r_ccdf)
    assert_allclose(p_ppcf, r_ppcf)


def test_rotation_180_deg():
    """Tests the 180° copula rotation."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Clayton copula family rotated 180°
    copula = ClaytonCopula(5, rotation='180°')
    # Comparison values
    r_logpdf = np.array([-np.inf, 0.6666753203, 0.9946292379, 0.7858645247,
                         -np.inf])
    r_pdf = np.array([0.0, 1.9477508961, 2.7037217178, 2.1943031503, 0.0])
    r_logcdf = np.array([-np.inf, -1.5602819348, -0.8286269453,
                         -0.4437013452, -0.2231435513])
    r_cdf = np.array([0.0, 0.2100768349, 0.4366484171, 0.6416570262, 0.8])
    r_ccdf = np.array([0.0, 0.3163606244, 0.5564206557, 0.8916601339, 1.0])
    r_ppcf = np.array([0.0, 0.2140692068, 0.4788690972, 0.7005153398, 1.0])
    p_logpdf = copula.logpdf(samples)
    p_pdf = copula.pdf(samples)
    p_logcdf = copula.logcdf(samples)
    p_cdf = copula.cdf(samples)
    p_ccdf = copula.ccdf(samples)
    p_ppcf = copula.ppcf(samples)
    assert_allclose(p_logpdf, r_logpdf)
    assert_allclose(p_pdf, r_pdf)
    assert_allclose(p_logcdf, r_logcdf)
    assert_allclose(p_cdf, r_cdf)
    assert_allclose(p_ccdf, r_ccdf)
    assert_allclose(p_ppcf, r_ppcf)


def test_rotation_270_deg():
    """Tests the 270° copula rotation."""
    samples = np.array([np.linspace(0, 1, 5), np.linspace(0.2, 0.8, 5)]).T

    # Clayton copula family rotated 270°
    copula = ClaytonCopula(5, rotation='270°')
    # Comparison values
    r_logpdf = np.array([-np.inf, -1.7680858282, 0.9946292379,
                         -2.5713221880, -np.inf])
    r_pdf = np.array([0.0, 0.1706593477, 2.7037217178, 0.0764344179, 0.0])
    r_logcdf = np.array([-38.123095, -6.7509119186, -2.7590553856,
                         -0.9153652928, -0.2231435513])
    r_cdf = np.array([0.0, 0.0011698124, 0.0633515829, 0.400370347, 0.8])
    r_ccdf = np.array([0.0, 0.0198871044, 0.5564206557, 0.996791538, 1.0])
    r_ppcf = np.array([0.0, 0.547275029, 0.4788690972, 0.4493004266, 1.0])
    p_logpdf = copula.logpdf(samples)
    p_pdf = copula.pdf(samples)
    p_logcdf = copula.logcdf(samples)
    p_cdf = copula.cdf(samples)
    p_ccdf = copula.ccdf(samples)
    p_ppcf = copula.ppcf(samples)
    assert_allclose(p_logpdf, r_logpdf)
    assert_allclose(p_pdf, r_pdf)
    assert_allclose(p_logcdf, r_logcdf)
    assert_allclose(p_cdf, r_cdf, atol=1e-10)
    assert_allclose(p_ccdf, r_ccdf, atol=1e-10)
    assert_allclose(p_ppcf, r_ppcf, atol=1e-10)
