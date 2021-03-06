# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from gala.units import galactic
from gala.potential.potential.cpotential cimport CPotentialWrapper
from gala.potential.potential.cpotential import CPotentialBase

cdef extern from "src/funcdefs.h":
    ctypedef double (*densityfunc)(double t, double *pars, double *q, int n_dim, int n_dim) nogil
    ctypedef double (*energyfunc)(double t, double *pars, double *q, int n_dim, int n_dim) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, int n_dim, int n_dim, double *grad) nogil
    ctypedef void (*hessianfunc)(double t, double *pars, double *q, int n_dim, int n_dim, double *hess) nogil

cdef extern from "potential/src/cpotential.h":
    enum:
        MAX_N_COMPONENTS = 16

    ctypedef struct CPotential:
        int n_components
        int n_dim
        densityfunc density[MAX_N_COMPONENTS]
        energyfunc value[MAX_N_COMPONENTS]
        gradientfunc gradient[MAX_N_COMPONENTS]
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]

cdef extern from "src/_potential.h":
    double wang_zhao_bar_value(double t, double *pars, double *q, int n_dim) nogil
    void wang_zhao_bar_gradient(double t, double *pars, double *q, int n_dim, double *grad) nogil
    double wang_zhao_bar_density(double t, double *pars, double *q, int n_dim) nogil

cdef class WangZhaoBarWrapper(CPotentialWrapper):

    def __init__(self, G, parameters, q0):
        self.init([G] + list(parameters), np.ascontiguousarray(q0))
        self.cpotential.value[0] = <energyfunc>(wang_zhao_bar_value)
        self.cpotential.density[0] = <densityfunc>(wang_zhao_bar_density)
        self.cpotential.gradient[0] = <gradientfunc>(wang_zhao_bar_gradient)

class WangZhaoBarPotential(CPotentialBase):
    r"""
    WangZhaoBarPotential(m, r_s, alpha, Omega, units)

    TODO:

    Parameters
    ----------
    TODO
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.

    """
    def __init__(self, m, r_s, alpha, Omega, units=galactic):
        parameters = OrderedDict()
        parameters['m'] = m
        parameters['r_s'] = r_s
        parameters['alpha'] = alpha
        parameters['Omega'] = Omega

        super(WangZhaoBarPotential, self).__init__(parameters=parameters,
                                                   units=units,
                                                   Wrapper=WangZhaoBarWrapper)
