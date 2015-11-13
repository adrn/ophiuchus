# coding: utf-8

""" Astropy coordinate class for the Ophiuchus coordinate system """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

from astropy.coordinates import frame_transform_graph
from astropy.utils.data import get_pkg_data_filename
import astropy.coordinates as coord
import astropy.units as u

__all__ = ["Ophiuchus"]

class Ophiuchus(coord.BaseCoordinateFrame):
    """
    A Heliocentric spherical coordinate system defined by the orbit
    of the Ophiuchus stream.

    For more information about how to use this class, see the Astropy documentation
    on `Coordinate Frames <http://docs.astropy.org/en/latest/coordinates/frames.html>`_.

    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    phi1 : `Angle`, optional, must be keyword
        The longitude-like angle corresponding to the orbit.
    phi2 : `Angle`, optional, must be keyword
        The latitude-like angle corresponding to the orbit.
    distance : `Quantity`, optional, must be keyword
        The Distance for this object along the line-of-sight.

    """
    default_representation = coord.SphericalRepresentation

    frame_specific_representation_info = {
        'spherical': [coord.RepresentationMapping('lon', 'phi1'),
                      coord.RepresentationMapping('lat', 'phi2'),
                      coord.RepresentationMapping('distance', 'distance')],
        'unitspherical': [coord.RepresentationMapping('lon', 'phi1'),
                          coord.RepresentationMapping('lat', 'phi2')]
    }

# read the rotation matrix (previously generated)
R = np.loadtxt(get_pkg_data_filename('rotationmatrix.txt'))

# Galactic to Ophiuchus coordinates
@frame_transform_graph.transform(coord.FunctionTransform, coord.Galactic, Ophiuchus)
def galactic_to_oph(gal_coord, oph_frame):
    """ Compute the transformation from Galactic spherical to
        heliocentric Oph coordinates.
    """

    l = np.atleast_1d(gal_coord.l.radian)
    b = np.atleast_1d(gal_coord.b.radian)

    X = np.cos(b)*np.cos(l)
    Y = np.cos(b)*np.sin(l)
    Z = np.sin(b)

    # Calculate X,Y,Z,distance in the Oph system
    Xs, Ys, Zs = R.dot(np.array([X, Y, Z]))

    # Calculate the angular coordinates lambda,beta
    Lambda = np.arctan2(Ys, Xs)*u.radian
    Lambda[Lambda < 0] = Lambda[Lambda < 0] + 2.*np.pi*u.radian
    Beta = np.arcsin(Zs/np.sqrt(Xs*Xs+Ys*Ys+Zs*Zs))*u.radian

    return Ophiuchus(phi1=Lambda, phi2=Beta,
                     distance=gal_coord.distance)


# Oph to Galactic coordinates
@frame_transform_graph.transform(coord.FunctionTransform, Ophiuchus, coord.Galactic)
def oph_to_galactic(oph_coord, gal_frame):
    """ Compute the transformation from heliocentric Oph coordinates to
        spherical Galactic.
    """
    L = np.atleast_1d(oph_coord.Lambda.radian)
    B = np.atleast_1d(oph_coord.Beta.radian)

    Xs = np.cos(B)*np.cos(L)
    Ys = np.cos(B)*np.sin(L)
    Zs = np.sin(B)

    X, Y, Z = R.T.dot(np.array([Xs, Ys, Zs]))

    l = np.arctan2(Y, X)*u.radian
    b = np.arcsin(Z/np.sqrt(X*X+Y*Y+Z*Z))*u.radian

    l[l<0] += 2*np.pi*u.radian

    return coord.Galactic(l=l, b=b, distance=oph_coord.distance)
