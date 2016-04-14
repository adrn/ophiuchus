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
from gary.units import galactic
from gary.potential.cpotential cimport CPotentialWrapper
from gary.potential.cpotential import CPotentialBase

cdef extern from "src/cpotential.h":
    enum:
        MAX_N_COMPONENTS = 16

    ctypedef double (*densityfunc)(double t, double *pars, double *q) nogil
    ctypedef double (*valuefunc)(double t, double *pars, double *q) nogil
    ctypedef void (*gradientfunc)(double t, double *pars, double *q, double *grad) nogil

    ctypedef struct CPotential:
        int n_components
        int n_dim
        densityfunc density[MAX_N_COMPONENTS]
        valuefunc value[MAX_N_COMPONENTS]
        gradientfunc gradient[MAX_N_COMPONENTS]
        int n_params[MAX_N_COMPONENTS]
        double *parameters[MAX_N_COMPONENTS]

cdef extern from "src/_potential.h":
    double wang_zhao_bar_value(double t, double *pars, double *q) nogil
    void wang_zhao_bar_gradient(double t, double *pars, double *q, double *grad) nogil
    double wang_zhao_bar_density(double t, double *pars, double *q) nogil

cdef class WangZhaoBarWrapper(CPotentialWrapper):

    def __init__(self, G, parameters):
        cdef CPotential cp

        # This is the only code that needs to change per-potential
        cp.value[0] = <valuefunc>(wang_zhao_bar_value)
        cp.density[0] = <densityfunc>(wang_zhao_bar_density)
        cp.gradient[0] = <gradientfunc>(wang_zhao_bar_gradient)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        cp.n_components = 1
        self._params = np.array([G] + list(parameters), dtype=np.float64)
        self._n_params = np.array([len(self._params)], dtype=np.int32)
        cp.n_params = &(self._n_params[0])
        cp.parameters[0] = &(self._params[0])
        cp.n_dim = 3
        self.cpotential = cp

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
