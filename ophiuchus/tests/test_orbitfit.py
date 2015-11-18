# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import matplotlib.pyplot as pl
import numpy as np
import gary.potential as gp
import gary.integrate as gi

# Project
from .. import potential as op
from ..data import OphiuchusData
from ..orbitfit import ln_likelihood, ln_posterior
from ..plot import plot_data_orbit

def test_optimize():
    all_ophdata = OphiuchusData()
    fit_ophdata = OphiuchusData(expr="source=='Sesar2015a'") # only fit thin part
    potential = gp.load("/Users/adrian/projects/ophiuchus/potentials/static_mw.yml", module=op)
    # freeze = dict(t_forw=3., t_back=-5.,
    #               phi2_sigma=np.radians(0.1),
    #               d_sigma=0.025,
    #               vr_sigma=0.002)
    freeze = dict(phi2_sigma=np.radians(0.1),
                  d_sigma=0.025,
                  vr_sigma=0.002)

    dt = 0.5
    args = (fit_ophdata, potential, dt, freeze)
    guess = np.array([-2.70234950e-03, 8.44, -3.40761632e-02, 3.15424900e-03, 2.91923172e-01, 2.75, -8.])

    ll = ln_likelihood(guess, *args)
    print(ll)

    import scipy.optimize as so
    res = so.minimize(lambda *args,**kwargs: -ln_posterior(*args,**kwargs),
                      x0=guess, method='powell', args=args)
    best = res.x
    print(res)

    t_forw,t_back = best[-2:]
    best_w0 = fit_ophdata._mcmc_sample_to_w0(best[:-2])[:,0]
    t,w1 = potential.integrate_orbit(best_w0, dt=-dt, t1=0., t2=t_back, Integrator=gi.DOPRI853Integrator)
    t,w2 = potential.integrate_orbit(best_w0, dt=dt, t1=0., t2=t_forw, Integrator=gi.DOPRI853Integrator)
    w = np.vstack((w1[::-1,0], w2[1:,0]))
    fig = plot_data_orbit(all_ophdata, orbit_w=w)
    pl.show()

def test_convex():
    # check that the likelihood is convex along each 1D slice
    ophdata = OphiuchusData()
    potential = gp.load("/Users/adrian/projects/ophiuchus/potentials/static_mw.yml", module=op)
    freeze = dict(phi2_sigma=np.radians(0.1),
                  d_sigma=0.025,
                  vr_sigma=0.002)

    dt = 0.5
    args = (ophdata, potential, dt, freeze)
    guess = np.array([-2.70234950e-03, 8.44, -3.40761632e-02, 3.15424900e-03, 2.91923172e-01, 2.75, -8.])

    facs = np.linspace(1/1.2,1.2,64)
    # facs = np.linspace(1/1.03,1/1.021,8)
    # for i in [1]:
    for i in range(guess.size):
        g = guess.copy()
        lls = []
        orbits = [] # TESTING
        for fac in facs:
            g[i] = fac*guess[i]
            ll = ln_likelihood(g, *args)
            lls.append(ll.sum())

            guess_w0 = ophdata._mcmc_sample_to_w0(g)[:,0]
            t,w1 = potential.integrate_orbit(guess_w0, dt=-dt, t1=0., t2=-8, Integrator=gi.DOPRI853Integrator)
            t,w2 = potential.integrate_orbit(guess_w0, dt=dt, t1=0., t2=2.75, Integrator=gi.DOPRI853Integrator)
            w = np.vstack((w1[::-1,0], w2[1:,0]))
            orbits.append(w)

        fig,axes = pl.subplots(1,2,figsize=(10,4))
        fig.suptitle(str(i), fontsize=20)
        axes[0].plot(facs*guess[i], lls)
        axes[1].plot(facs*guess[i], np.exp(np.array(lls)-max(lls)))

        # for orbit,ll in zip(orbits,lls):
        #     fig = plot_data_orbit(ophdata, orbit_w=orbit)
        #     fig.suptitle("{:.2e}".format(ll))

    pl.show()
