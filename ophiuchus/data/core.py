# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
from astropy import log as logger
import astropy.coordinates as coord
import astropy.units as u
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import numexpr

import gary.coordinates as gc
from gary.dynamics import orbitfit
from gary.observation import distance
from gary.units import galactic
from gary.util import atleast_2d

# Project
from .. import galactocentric_frame, vcirc, vlsr

__all__ = ['OphiuchusData']

class OphiuchusData(object):
    """
    Utility class for interacting with the data for the Ophiuchus stream.
    """
    def __init__(self, expr=None):
        # read the catalog data file
        filename = get_pkg_data_filename('sesar.txt')
        _tbl = np.genfromtxt(filename, dtype=None, skip_header=2, names=True)
        if expr is not None:
            ix = numexpr.evaluate(expr, _tbl)
            _tbl = _tbl[ix]

        # convert distance modulus uncertainty to distance uncertainty
        dists = []
        dist_errs = []
        for DM,err_DM in zip(_tbl['DM'], _tbl['err_DM']):
            d = distance(np.random.normal(DM, err_DM, size=1024)).to(u.kpc).value
            dists.append(np.median(d))
            dist_errs.append(np.std(d))
        dists = np.array(dists)*u.kpc
        dist_errs = np.array(dist_errs)*u.kpc

        # make an astropy coordinate object from the positions
        self.coord = coord.ICRS(ra=_tbl['ra']*u.degree, dec=_tbl['dec']*u.degree, distance=dists)\
                          .transform_to(coord.Galactic)
        self.coord_err = dict(
            l=0.*self.coord.l.decompose(galactic),
            b=0.*self.coord.l.decompose(galactic),
            distance=dist_errs.decompose(galactic)
        )

        # compute the rotation matrix to go from Galacic to Stream coordinates by assuming
        #   the star with maximum stream longitude is the pivot
        self.R = orbitfit.compute_stream_rotation_matrix(self.coord, align_lon='max')

        # a SphericalRepresentation of the coordinates in Ophiuchus coordinates
        self.coord_oph = orbitfit.rotate_sph_coordinate(self.coord, self.R)

        # velocity information and uncertainties
        self.veloc = dict(
            mul=(_tbl['mu_l']*u.mas/u.yr).decompose(galactic),
            mub=(_tbl['mu_b']*u.mas/u.yr).decompose(galactic),
            vr=(_tbl['v_los']*u.km/u.s).decompose(galactic)
        )
        self.veloc_err = dict(
            mul=(_tbl['err_mu_l']*u.mas/u.yr).decompose(galactic),
            mub=(_tbl['err_mu_b']*u.mas/u.yr).decompose(galactic),
            vr=(_tbl['err_v_los']*u.km/u.s).decompose(galactic)
        )

    def oph_to_galactic(self, rep):
        """
        Transform from Ophiuchus stream coordinates to Galactic coordinates.
        """
        xyz = rep.represent_as(coord.CartesianRepresentation).xyz.value
        in_frame_car = coord.CartesianRepresentation(self.R.T.dot(xyz).T*u.kpc)
        return self.coord.realize_frame(in_frame_car)

    def _mcmc_sample_to_coord(self, p):
        p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
        rep = coord.SphericalRepresentation(lon=p[0]*0.*u.radian, # this is required by the MCMC
                                            lat=p[0]*u.radian, # this index looks weird but is right
                                            distance=p[1]*u.kpc)
        return self.oph_to_galactic(rep)

    def _mcmc_sample_to_w0(self, p):
        p = atleast_2d(p, insert_axis=-1) # note: from Gary, not Numpy
        c = self._mcmc_sample_to_coord(p)
        x0 = c.transform_to(galactocentric_frame).cartesian.xyz.decompose(galactic).value
        v0 = gc.vhel_to_gal(c, pm=(p[2]*u.rad/u.Myr,p[3]*u.rad/u.Myr), rv=p[4]*u.kpc/u.Myr,
                            galactocentric_frame=galactocentric_frame,
                            vcirc=vcirc, vlsr=vlsr).decompose(galactic).value
        w0 = np.concatenate((x0, v0))
        return w0
