# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np

import gary.potential as gp
import gary.dynamics as gd
from gary.units import galactic

# Project
from ..util import integrate_forward_backward

def test_integrate_forward_backward():
    pot = gp.SphericalNFWPotential(v_c=150*u.km/u.s, r_s=10*u.kpc, units=galactic)

    # with array
    w0 = [15.,0,0,0,0.12,0]
    orbit = integrate_forward_backward(pot, w0, 100., -200., dt=1.)
    assert orbit.pos.shape == (3,300)
