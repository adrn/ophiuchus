# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import gary.integrate as gi
from gary.dynamics.orbit import combine

__all__ = ['integrate_forward_backward']

def integrate_forward_backward(potential, w0, t_forw, t_back, dt=0.5,
                               Integrator=gi.DOPRI853Integrator):
    """
    TODO:
    """

    o1 = potential.integrate_orbit(w0, dt=-dt, t1=0., t2=t_back, Integrator=Integrator)
    o2 = potential.integrate_orbit(w0, dt=dt, t1=0., t2=t_forw, Integrator=Integrator)

    o1 = o1[::-1]
    o2 = o2[1:]
    orbit = combine((o1, o2), along_time_axis=True)

    if orbit.pos.shape[-1] == 1:
        return orbit[:,0]
    else:
        return orbit
