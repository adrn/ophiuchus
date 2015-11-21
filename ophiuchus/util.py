# coding: utf-8

""" General utilities. """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
import gary.integrate as gi

__all__ = ['integrate_forward_backward']

def integrate_forward_backward(potential, w0, t_forw, t_back, dt=0.5):
    """
    TODO:
    """

    t,w1 = potential.integrate_orbit(w0, dt=-dt, t1=0., t2=t_back,
                                     Integrator=gi.DOPRI853Integrator)
    t,w2 = potential.integrate_orbit(w0, dt=dt, t1=0., t2=t_forw,
                                     Integrator=gi.DOPRI853Integrator)
    w_out = np.vstack((w1[::-1], w2[1:]))
    if w_out.shape[1] == 1:
        return w_out[:,0]
    else:
        return w_out
