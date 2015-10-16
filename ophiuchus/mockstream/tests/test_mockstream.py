# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import matplotlib.pyplot as pl
import numpy as np

# Custom
import gary.dynamics as gd
import gary.integrate as gi
import gary.potential as gp
from gary.units import galactic

# Project
from .._mockstream import streakline_stream

def test_streakline():
    potential = gp.SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic)
    release_every = 1
    time = 1024
    dt = 2
    nsteps = time // dt

    # w0 = np.array([0.,15.,0,-0.2,0,0]) # circular
    w0 = np.array([0.,15.,0,-0.15,0,0]) # eccentric

    print("integrating parent orbit")
    torig,w = potential.integrate_orbit(w0, dt=-dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    t,w = potential.integrate_orbit(w[-1], dt=dt, t1=torig[-1], nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    ww = w[:,0].copy()
    print("done")

    # fig = gd.plot_orbits(w, marker=None)
    # pl.show()

    prog_mass = np.zeros_like(t) + 1E4

    stream = streakline_stream(potential.c_instance, t, ww,
                               release_every=release_every, G=potential.G,
                               prog_mass=prog_mass)

    ixs = [0,1]
    pl.figure(figsize=(6,6))
    pl.plot(ww[:,ixs[0]], ww[:,ixs[1]], marker=None, alpha=0.5)
    pl.scatter(stream[::2,ixs[0]], stream[::2,ixs[1]], c=t[:-1:release_every], vmin=t.min(), vmax=t.max(), cmap='coolwarm', s=4)
    pl.scatter(stream[1::2,ixs[0]], stream[1::2,ixs[1]], c=t[:-1:release_every], vmin=t.min(), vmax=t.max(), cmap='coolwarm', s=4)
    pl.xlim(-0.6, 0.6)
    pl.ylim(14.9,15.1)
    pl.show()
