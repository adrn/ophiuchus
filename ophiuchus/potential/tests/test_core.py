# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import numpy as np

# Project
from .. import WangZhaoBarPotential

def test_wangzhao():
    p = WangZhaoBarPotential(1E10, 1., 27*u.degree, 60*u.km/u.s/u.kpc)
    print(p.value([4., 0, 0]))
    print(p.gradient([4., 0, 0]))
    print(p.density([4., 0, 0]).to(u.Msun/u.pc**3))
