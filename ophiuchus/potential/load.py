# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.utils.data import get_pkg_data_filename
import gary.potential as gp

# Project
from .. import potential as op

def load_potential(name):
    return gp.load(get_pkg_data_filename('{}.yml'.format(name)), module=op)

barred_mw = load_potential('barred_mw')
static_mw = load_potential('static_mw')
