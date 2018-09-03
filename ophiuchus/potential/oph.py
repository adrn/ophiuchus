# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from gala.units import galactic
from gala.potential import (CCompositePotential, MiyamotoNagaiPotential,
                            HernquistPotential, NFWPotential)

# Project
from . import WangZhaoBarPotential

class OphiuchusPotential(CCompositePotential):
    r"""
    Four-component Milky Way potential used for modeling the Ophiuchus stream.

    Parameters
    ----------
    units : iterable
        Unique list of non-reducable units that specify (at minimum) the
        length, mass, time, and angle units.
    spheroid : dict
        Dictionary of parameter values for a :class:`gala.potential.HernquistPotential`.
    disk : dict
        Dictionary of parameter values for a :class:`gala.potential.MiyamotoNagaiPotential`.
    halo : dict
        Dictionary of parameter values for a :class:`gala.potential.NFWPotential`.
    bar : dict
        Dictionary of parameter values for a :class:`ophiuchus.potential.WangZhaoBarPotential`.

    """
    def __init__(self, units=galactic,
                 spheroid=None, disk=None, halo=None, bar=None):
        default_spheroid = dict(m=0., c=0.1)
        default_disk = dict(m=5.E10, a=3, b=0.28) # similar to Bovy
        default_halo = dict(v_c=0.19, r_s=30., c=0.9)
        default_bar = dict(m=1.8E10 / 1.15, r_s=1.,
                           alpha=0.349065850398, Omega=0.06136272990322247) # from Wang, Zhao, et al.

        if disk is None:
            disk = default_disk
        else:
            for k, v in default_disk.items():
                if k not in disk:
                    disk[k] = v

        if spheroid is None:
            spheroid = default_spheroid
        else:
            for k, v in default_spheroid.items():
                if k not in spheroid:
                    spheroid[k] = v

        if halo is None:
            halo = default_halo
        else:
            for k, v in default_halo.items():
                if k not in halo:
                    halo[k] = v

        if bar is None:
            bar = default_bar
        else:
            for k, v in default_bar.items():
                if k not in bar:
                    bar[k] = v

        super(OphiuchusPotential, self).__init__()

        self["spheroid"] = HernquistPotential(units=units, **spheroid)
        self["disk"] = MiyamotoNagaiPotential(units=units, **disk)
        self["halo"] = NFWPotential(units=units, **halo)
        self["bar"] = WangZhaoBarPotential(units=units, **bar)
