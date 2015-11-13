# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
from gary.dynamics import orbitfit

# Project
from ..core import Ophiuchus, R

def test_roundtrip_transform():
    n = 128
    g = coord.SkyCoord(l=np.random.uniform(0,360,size=n)*u.deg,
                       b=np.random.uniform(-60,60,size=n)*u.deg,
                       distance=np.random.uniform(0,100,size=n)*u.kpc,
                       frame='galactic')
    o = g.transform_to(Ophiuchus)
    assert np.allclose(o.distance.value, g.distance.value)

    g2 = o.transform_to(coord.Galactic)
    assert np.allclose(g2.distance.value, g.distance.value)
    assert np.allclose(g2.l.value, g.l.value, atol=1E-9)
    assert np.allclose(g2.b.value, g.b.value, atol=1E-9)
