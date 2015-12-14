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

def ophiuchus_stream(potential, prog_orbit, prog_mass, t_disrupt,
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
    prog_orbit : `~gary.dynamics.Orbit`
        The orbit of the progenitor system.
    prog_mass : numeric, array_like
        A single mass or an array of masses if the progenitor mass evolves
        with time.
    t_disrupt : numeric
        The time that the progenitor completely disrupts.
    release_every : int (optional)
        Release particles at the Lagrange points every X timesteps.
    Integrator : `~gary.integrate.Integrator` (optional)
        Integrator to use.
    Integrator_kwargs : dict (optional)
        Any extra keyword argumets to pass to the integrator function.

    Returns
    -------
    stream : `~gary.dynamics.CartesianPhaseSpacePosition`

    """

    # the time index closest to when the disruption happens
    t = prog_orbit.t
    disrupt_ix = np.abs(t - t_disrupt).argmin()

    k_mean = np.zeros((t.size,6))
    k_disp = np.zeros((t.size,6))

    k_mean[:,0] = 2. # R
    k_mean[disrupt_ix:,0] = 0.
    k_disp[:,0] = 0.5

    k_mean[:,1] = 0. # phi
    k_disp[:,1] = 0.

    k_mean[:,2] = 0. # z
    k_disp[:,2] = 0.5

    k_mean[:,3] = 0. # vR
    k_disp[:,3] = 0.

    k_mean[:,4] = 0.3 # vt
    k_mean[disrupt_ix:,4] = 0.
    k_disp[:,4] = 0.5

    k_mean[:,5] = 0. # vz
    k_disp[:,5] = 0.5

    return mock_stream(potential=potential, prog_orbit=prog_orbit, prog_mass=prog_mass,
                       k_mean=k_mean, k_disp=k_disp, release_every=release_every,
                       Integrator=Integrator, Integrator_kwargs=Integrator_kwargs)
