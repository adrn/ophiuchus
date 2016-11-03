# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.utils.data import get_pkg_data_filename
import gala.potential as gp

def load_potential(name):
    from .. import potential as op
    return gp.load(get_pkg_data_filename('yml/{}.yml'.format(name)), module=op)

# barred_mw = load_potential('barred_mw')
# static_mw = load_potential('static_mw')
