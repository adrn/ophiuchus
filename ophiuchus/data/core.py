# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
from collections import OrderedDict

# Third-party
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import numexpr

import gary.coordinates as gc
from gary.observation import distance
from gary.units import galactic
from gary.util import atleast_2d

# Project
from .. import galactocentric_frame, vcirc, vlsr
from ..coordinates import Ophiuchus

__all__ = ['OphiuchusData']

class OphiuchusData(object):
    """
    Utility class for interacting with the data for the Ophiuchus stream.
    """
    def __init__(self, expr=None, index=None):
        # read the catalog data file
        filename = get_pkg_data_filename('sesar.txt')
        _tbl = np.genfromtxt(filename, dtype=None, skip_header=2, names=True)

        if expr is not None and index is not None:
            raise ValueError("Can't specify expr and index selection.")

        elif expr is not None:
            ix = numexpr.evaluate(expr, _tbl)
            _tbl = _tbl[ix]

        elif index is not None:
            _tbl = _tbl[index]

        # convert distance modulus uncertainty to distance uncertainty
        dists = []
        dist_errs = []
        for DM,err_DM in zip(_tbl['DM'], _tbl['err_DM']):
            d = distance(np.random.normal(DM, err_DM, size=1024)).to(u.kpc).value
            dists.append(distance(DM).to(u.kpc).value)
            dist_errs.append(np.std(d))
        dists = np.array(dists)*u.kpc
        dist_errs = np.array(dist_errs)*u.kpc

        # make an astropy coordinate object from the positions
        self.coord = coord.ICRS(ra=_tbl['ra']*u.degree, dec=_tbl['dec']*u.degree, distance=dists)\
                          .transform_to(coord.Galactic)
        self.coord_err = OrderedDict()
        self.coord_err['l'] = 0.*self.coord.l.decompose(galactic)
        self.coord_err['b'] = 0.*self.coord.l.decompose(galactic)
        self.coord_err['distance'] = dist_errs.decompose(galactic)

        # a SphericalRepresentation of the coordinates in Ophiuchus coordinates
        # self.coord_oph = orbitfit.rotate_sph_coordinate(self.coord, self.R)
        self.coord_oph = self.coord.transform_to(Ophiuchus)

        # velocity information and uncertainties
        self.veloc = OrderedDict()
        self.veloc['mul'] = (_tbl['mu_l']*u.mas/u.yr).decompose(galactic)
        self.veloc['mub'] = (_tbl['mu_b']*u.mas/u.yr).decompose(galactic)
        self.veloc['vr'] = (_tbl['v_los']*u.km/u.s).decompose(galactic)

        self.veloc_err = OrderedDict()
        self.veloc_err['mul'] = (_tbl['err_mu_l']*u.mas/u.yr).decompose(galactic)
        self.veloc_err['mub'] = (_tbl['err_mu_b']*u.mas/u.yr).decompose(galactic)
        self.veloc_err['vr'] = (_tbl['err_v_los']*u.km/u.s).decompose(galactic)

        self._tbl = _tbl

    def _mcmc_sample_to_coord(self, p):
        p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
        oph = Ophiuchus(phi1=p[0]*0.*u.radian, # this is required by the MCMC
                        phi2=p[0]*u.radian, # this index looks weird but is right
                        distance=p[1]*u.kpc)
        return oph.transform_to(coord.Galactic)

    def _mcmc_sample_to_w0(self, p):
        p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
        c = self._mcmc_sample_to_coord(p)
        x0 = c.transform_to(galactocentric_frame).cartesian.xyz.decompose(galactic).value
        v0 = gc.vhel_to_gal(c, pm=(p[2]*u.rad/u.Myr,p[3]*u.rad/u.Myr), rv=p[4]*u.kpc/u.Myr,
                            galactocentric_frame=galactocentric_frame,
                            vcirc=vcirc, vlsr=vlsr).decompose(galactic).value
        w0 = np.concatenate((x0, v0))
        return w0

    def _w0_to_gal_coord_veloc(self, w0):
        w0 = np.atleast_2d(w0)
        w_coord = galactocentric_frame.realize_frame(coord.CartesianRepresentation(w0.T[:3]*u.kpc))\
                                      .transform_to(coord.Galactic)
        w_vel = gc.vgal_to_hel(w_coord, w0.T[3:]*u.kpc/u.Myr,
                               galactocentric_frame=galactocentric_frame,
                               vcirc=vcirc, vlsr=vlsr)
        veloc = OrderedDict()
        veloc['mul'] = w_vel[0]
        veloc['mub'] = w_vel[1]
        veloc['vr'] = w_vel[2]
        return w_coord, veloc
