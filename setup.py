# -*- coding: utf-8 -*-
# Copyright (C) 2017-2019 Arno Onken
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
Setup for the mixedvines package.
'''
from setuptools import setup

setup(
    name="mixedvines",
    version="1.2.2",
    description=("Package for canonical vine copula trees with mixed"
                 " continuous and discrete marginals."),
    long_description=open("README.rst").read(),
    long_description_content_type="text/markdown",
    keywords="copula mixed vine continuous dicrete entropy",
    url="https://github.com/asnelt/mixedvines/",
    author="Arno Onken",
    author_email="asnelt@asnelt.org",
    license="GPLv3+",
    packages=["mixedvines"],
    install_requires=[
        "numpy",
        "scipy"],
    test_suite="nose2.collector.collector",
    tests_require=["nose2"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        ("License :: OSI Approved :: GNU General Public License v3 or later"
         " (GPLv3+)"),
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"]
)
