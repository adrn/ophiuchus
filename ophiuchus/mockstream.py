# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
import astropy.units as u
import numpy as np
from gary.dynamics.mockstream import mock_stream
from gary.integrate import LeapfrogIntegrator

def ophiuchus_stream(potential, w0, prog_mass, t_disrupt, t_f, dt=1., t_0=0.,
                     release_every=1, Integrator=LeapfrogIntegrator, Integrator_kwargs=dict()):
    """
    Generate a mock stellar stream in the specified potential with a
    progenitor system that ends up at the specified position.

    This uses the prescription used in Price-Whelan et al. (2016) for
    models of the Ophiuchus stream.

    Parameters
    ----------
    potential : `~gary.potential.PotentialBase`
        The gravitational potential.
    w0 : `~gary.dynamics.PhaseSpacePosition`, array_like
        Initial conditions.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    t_disrupt : numeric
        The time that the progenitor completely disrupts.
    t_f : numeric
        The final time for integrating.
    t_0 : numeric (optional)
        The initial time for integrating -- the time at which ``w0`` is the position.
    dt : numeric (optional)
        The time-step.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gary.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    prog_orbit : `~gary.dynamics.CartesianOrbit`
    stream : `~gary.dynamics.CartesianPhaseSpacePosition`

    """

    # the time index closest to when the disruption happens
    nsteps = int(round(np.abs((t_f-t_0)/dt)))
    t = np.linspace(t_0, t_f, nsteps)
    disrupt_ix = np.abs(t - t_disrupt).argmin()

    if dt < 0:
        s = slice(disrupt_ix, None)
    else:
        s = slice(None, disrupt_ix)

    k_mean = np.zeros((nsteps,6))
    k_disp = np.zeros((nsteps,6))

    k_mean[:,0] = 1. # R
    k_mean[s,0] = 0.
    k_disp[:,0] = 0.5

    k_mean[:,1] = 0. # phi
    k_disp[:,1] = 0.

    k_mean[:,2] = 0. # z
    k_disp[:,2] = 0.5

    k_mean[:,3] = 0. # vR
    k_disp[:,3] = 0.

    k_mean[:,4] = 0.3 # vt
    k_mean[s,4] = 0.3
    k_disp[:,4] = 0.5

    k_mean[:,5] = 0. # vz
    k_disp[:,5] = 0.5

    return mock_stream(potential=potential, w0=w0, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp,
                       t_f=t_f, dt=dt, t_0=t_0, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)
