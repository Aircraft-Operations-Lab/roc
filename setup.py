#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='roc3',
      version='1.0',
      description='Robust Optimal Control for Flight Planning',
      packages=['roc3', 'roc3.occopy'],
      zip_safe=False, install_requires=['matplotlib', 'casadi==3.6.3', 'numpy', 'PyYAML', 'xarray', 'scipy', 'openpyxl'])
