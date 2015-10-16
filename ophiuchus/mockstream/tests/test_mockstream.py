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
from .._mockstream import streakline_stream, fardal_stream

def test_streakline():
    # potential = gp.LogarithmicPotential(v_c=0.2, r_h=20., q1=1., q2=1., q3=1., units=galactic)
    # potential = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.26, units=galactic)
    potential = gp.SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic)
    release_every = 1
    time = 2048
    dt = 2
    nsteps = time // dt

    # w0 = np.array([0.,15.,0,-0.2,0,0]) # circular
    w0 = np.array([0.,15.,0,-0.13,0,0]) # eccentric

    print("integrating parent orbit")
    torig,w = potential.integrate_orbit(w0, dt=-dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    t,w = potential.integrate_orbit(w[-1], dt=dt, t1=torig[-1], nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    ww = w[:,0].copy()
    print("done")
    r = np.sqrt(np.sum(ww[:,:3]**2, axis=-1))
    ecc = (r.max() - r.min()) / (r.min() + r.max())
    print("eccentricity", ecc)

    fig = gd.plot_orbits(w, marker=None)
    # pl.show()
    # return

    prog_mass = np.zeros_like(t) + 1E4

    stream = streakline_stream(potential.c_instance, t, ww,
                               release_every=release_every, G=potential.G,
                               prog_mass=prog_mass)

    ixs = [0,1]
    pl.figure(figsize=(6,6))
    pl.plot(ww[:,ixs[0]], ww[:,ixs[1]], marker=None, alpha=0.5)
    pl.scatter(stream[::2,ixs[0]], stream[::2,ixs[1]], c=t[:-1:release_every], vmin=t.min(), vmax=t.max(), cmap='coolwarm', s=4)
    pl.scatter(stream[1::2,ixs[0]], stream[1::2,ixs[1]], c=t[:-1:release_every], vmin=t.min(), vmax=t.max(), cmap='coolwarm', s=4)
    # pl.xlim(-0.6, 0.6)
    # pl.ylim(14.9,15.1)
    pl.show()

def test_fardal():
    # potential = gp.LogarithmicPotential(v_c=0.2, r_h=20., q1=1., q2=1., q3=1., units=galactic)
    # potential = gp.MiyamotoNagaiPotential(m=1E11, a=6.5, b=0.26, units=galactic)
    potential = gp.SphericalNFWPotential(v_c=0.2, r_s=20., units=galactic)
    release_every = 1
    time = 2048
    dt = 2
    nsteps = time // dt

    # w0 = np.array([0.,15.,0,-0.2,0,0]) # circular
    w0 = np.array([0.,15.,0,-0.13,0,0]) # eccentric

    print("integrating parent orbit")
    torig,w = potential.integrate_orbit(w0, dt=-dt, nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    t,w = potential.integrate_orbit(w[-1], dt=dt, t1=torig[-1], nsteps=nsteps, Integrator=gi.DOPRI853Integrator)
    ww = w[:,0].copy()
    print("done")
    r = np.sqrt(np.sum(ww[:,:3]**2, axis=-1))
    ecc = (r.max() - r.min()) / (r.min() + r.max())
    print("eccentricity", ecc)

    fig = gd.plot_orbits(w, marker=None)
    # pl.show()
    # return

    prog_mass = np.zeros_like(t) + 1E4

    stream = fardal_stream(potential.c_instance, t, ww,
                           release_every=release_every, G=potential.G,
                           prog_mass=prog_mass)

    ixs = [0,1]
    pl.figure(figsize=(6,6))
    pl.plot(ww[:,ixs[0]], ww[:,ixs[1]], marker=None, alpha=0.5)
    pl.scatter(stream[::2,ixs[0]], stream[::2,ixs[1]], c=t[:-1:release_every], vmin=t.min(), vmax=t.max(), cmap='coolwarm', s=4)
    pl.scatter(stream[1::2,ixs[0]], stream[1::2,ixs[1]], c=t[:-1:release_every], vmin=t.min(), vmax=t.max(), cmap='coolwarm', s=4)
    # pl.xlim(-0.6, 0.6)
    # pl.ylim(14.9,15.1)
    pl.show()