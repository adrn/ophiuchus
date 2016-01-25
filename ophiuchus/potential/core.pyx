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
from astropy.coordinates.angles import rotation_matrix
from astropy.constants import G
import astropy.units as u
import numpy as np
cimport numpy as np
np.import_array()
import cython
cimport cython

# Project
from gary.units import galactic
from gary.potential.cpotential cimport _CPotential
from gary.potential.cpotential import CPotentialBase

cdef extern from "src/_potential.h":
    double wang_zhao_bar_value(double t, double *pars, double *q) nogil
    void wang_zhao_bar_gradient(double t, double *pars, double *q, double *grad) nogil
    double wang_zhao_bar_density(double t, double *pars, double *q) nogil

    double ophiuchus_value(double t, double *pars, double *q) nogil
    void ophiuchus_gradient(double t, double *pars, double *q, double *grad) nogil
    double ophiuchus_density(double t, double *pars, double *q) nogil

cdef class _WangZhaoBarPotential(_CPotential):
    def __cinit__(self, double G, double m, double r_s, double alpha, double Omega):
        self._parvec = np.array([G,m,r_s,alpha,Omega])
        self._parameters = &(self._parvec[0])
        self.c_value = &wang_zhao_bar_value
        self.c_gradient = &wang_zhao_bar_gradient
        self.c_density = &wang_zhao_bar_density

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
        self.G = G.decompose(units).value
        self.parameters = dict()
        self.parameters['m'] = m
        self.parameters['r_s'] = r_s
        self.parameters['alpha'] = alpha
        self.parameters['Omega'] = Omega
        super(WangZhaoBarPotential, self).__init__(units=units)

        c_params = dict()
        c_params['G'] = self.G
        c_params['m'] = m
        c_params['r_s'] = r_s
        c_params['alpha'] = alpha
        c_params['Omega'] = Omega

        self.c_instance = _WangZhaoBarPotential(**c_params)

cdef class _OphiuchusPotential(_CPotential):

    def __cinit__(self, double G, double m_spher, double c,
                  double G2, double m_disk, double a, double b,
                  double G3, double v_c, double r_s, double q_z,
                  double G4, double m_bar, double a_bar, double alpha, double Omega
                  ):
        # alpha = initial bar angle
        # Omega = pattern speed
        self._parvec = np.array([G, m_spher, c, # 0,1,2
                                 G2, m_disk, a, b, # 3,4,5,6
                                 G3, v_c, r_s, q_z, # 7,8,9,10
                                 G4, m_bar, a_bar, alpha, Omega]) # 11,12,13,14,15
        self._parameters = &(self._parvec[0])
        self.c_value = &ophiuchus_value
        self.c_gradient = &ophiuchus_gradient
        self.c_density = &ophiuchus_density

class OphiuchusPotential(CPotentialBase):
    r"""
    OphiuchusPotential(units, spheroid=dict(), disk=dict(), halo=dict(), bar=dict())

    Four-component Milky Way potential used for modeling the Ophiuchus stream.

    Parameters
    ----------
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    spheroid : dict
        Dictionary of parameter values for a :class:`gary.potential.HernquistPotential`.
    disk : dict
        Dictionary of parameter values for a :class:`gary.potential.MiyamotoNagaiPotential`.
    halo : dict
        Dictionary of parameter values for a :class:`gary.potential.FlattenedNFWPotential`.
    bar : dict
        Dictionary of parameter values for a :class:`ophiuchus.potential.WangZhaoBarPotential`.

    """
    def __init__(self, units=galactic, spheroid=dict(), disk=dict(), halo=dict(), bar=dict()):
        self.G = G.decompose(units).value
        self.parameters = dict()
        default_spheroid = dict(m=0., c=0.1)
        default_disk = dict(m=5.E10, a=3, b=0.28) # similar to Bovy
        default_halo = dict(v_c=0.19, r_s=30., q_z=0.9)
        default_bar = dict(m=1.8E10 / 1.15, r_s=1.,
                           alpha=0.349065850398, Omega=0.06136272990322247) # from Wang, Zhao, et al.

        for k,v in default_disk.items():
            if k not in disk:
                disk[k] = v
        self.parameters['disk'] = disk

        for k,v in default_spheroid.items():
            if k not in spheroid:
                spheroid[k] = v
        self.parameters['spheroid'] = spheroid

        for k,v in default_halo.items():
            if k not in halo:
                halo[k] = v
        self.parameters['halo'] = halo

        for k,v in default_bar.items():
            if k not in bar:
                bar[k] = v
        self.parameters['bar'] = bar

        super(OphiuchusPotential, self).__init__(units=units)

        for name,group in self.parameters.items():
            for k,v in group.items():
                try:
                    group[k] = v.decompose(units).value
                except AttributeError:
                    pass

        c_params = dict()

        # bulge
        c_params['G'] = self.G
        c_params['m_spher'] = spheroid['m']
        c_params['c'] = spheroid['c']

        # disk
        c_params['G2'] = self.G
        c_params['m_disk'] = disk['m']
        c_params['a'] = disk['a']
        c_params['b'] = disk['b']

        # halo
        c_params['G3'] = self.G
        c_params['v_c'] = halo['v_c']
        c_params['r_s'] = halo['r_s']
        c_params['q_z'] = halo['q_z']

        # bar
        c_params['G4'] = self.G
        c_params['m_bar'] = bar['m']
        c_params['a_bar'] = bar['r_s']
        c_params['alpha'] = bar['alpha']
        c_params['Omega'] = bar['Omega']

        self.c_instance = _OphiuchusPotential(**c_params)
