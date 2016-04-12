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
from ..load import load_potential

def test_load():
    p = load_potential('barred_mw')
