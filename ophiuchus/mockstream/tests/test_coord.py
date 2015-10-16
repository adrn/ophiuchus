# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys
import logging

# Third-party
from astropy import log as logger
import matplotlib.pyplot as pl
import numpy as np

# Project
from .._coord import _test_car_to_cyl_roundtrip, _test_cyl_to_car_roundtrip

def test_car_to_cyl_roundtrip():
    _test_car_to_cyl_roundtrip()

def test_cyl_to_car_roundtrip():
    _test_cyl_to_car_roundtrip()
