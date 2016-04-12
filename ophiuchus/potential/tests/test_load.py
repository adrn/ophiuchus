# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy.tests.helper import quantity_allclose

# Project
from ..load import load_potential

def test_load():
    p_bar = load_potential('barred_mw')
    p_static = load_potential('static_mw')

    x = [4.1, -0.5, 0.1]
    assert not quantity_allclose(p_bar.value(x), p_static.value(x))
    assert not quantity_allclose(p_bar.gradient(x), p_static.gradient(x))
