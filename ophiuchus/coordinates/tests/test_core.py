# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import astropy.units as u
import astropy.coordinates as coord
import numpy as np
import gala.dynamics as gd

# Project
from ..core import Ophiuchus
from ...data import OphiuchusData

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

def test_data_phi2_is_small():
    d = OphiuchusData(expr="source == b'Sesar2015a'")
    oph = d.coord.transform_to(Ophiuchus)
    assert np.all(np.abs(oph.phi2) < 5.*u.arcmin) # all phi2 should be <5 arcmin

def test_orbit_transform():
    pos = np.random.uniform(size=(3,128))*u.kpc
    vel = np.random.uniform(size=(3,128))*u.kpc/u.Myr
    orbit = gd.CartesianOrbit(pos=pos, vel=vel)
    c,v = orbit.to_frame(coord.Galactic)
    oph = c.transform_to(Ophiuchus)
    pm_l,pm_b,vr = v

    assert pm_l.unit == u.mas/u.yr
    assert pm_b.unit == u.mas/u.yr
    assert vr.unit == vel.unit
