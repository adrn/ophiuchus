# coding: utf-8

""" Astropy coordinate class for the Sagittarius coordinate system """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np

from astropy.coordinates import frame_transform_graph
from astropy.coordinates.angles import rotation_matrix
import astropy.coordinates as coord
import astropy.units as u

__all__ = ["Orphan"]

class Orphan(coord.BaseCoordinateFrame):
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

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(phi, "z", unit=u.radian)
C = rotation_matrix(theta, "x", unit=u.radian)
B = rotation_matrix(psi, "z", unit=u.radian)
sgr_matrix = np.array(B.dot(C).dot(D))

# Galactic to Sgr coordinates
@frame_transform_graph.transform(coord.FunctionTransform, coord.Galactic, Orphan)
def galactic_to_orp(gal_coord, sgr_frame):
    """ Compute the transformation from Galactic spherical to
        heliocentric Sgr coordinates.
    """

    l = np.atleast_1d(gal_coord.l.radian)
    b = np.atleast_1d(gal_coord.b.radian)

    X = np.cos(b)*np.cos(l)
    Y = np.cos(b)*np.sin(l)
    Z = np.sin(b)

    # Calculate X,Y,Z,distance in the Sgr system
    Xs, Ys, Zs = sgr_matrix.dot(np.array([X, Y, Z]))

    # Calculate the angular coordinates lambda,beta
    Lambda = np.arctan2(Ys, Xs)*u.radian
    Lambda[Lambda < 0] = Lambda[Lambda < 0] + 2.*np.pi*u.radian
    Beta = np.arcsin(Zs/np.sqrt(Xs*Xs+Ys*Ys+Zs*Zs))*u.radian

    return Orphan(Lambda=Lambda, Beta=Beta,
                  distance=gal_coord.distance)


# Sgr to Galactic coordinates
@frame_transform_graph.transform(coord.FunctionTransform, Orphan, coord.Galactic)
def orp_to_galactic(orp_coord, gal_frame):
    """ Compute the transformation from heliocentric Sgr coordinates to
        spherical Galactic.
    """
    L = np.atleast_1d(orp_coord.Lambda.radian)
    B = np.atleast_1d(orp_coord.Beta.radian)

    Xs = np.cos(B)*np.cos(L)
    Ys = np.cos(B)*np.sin(L)
    Zs = np.sin(B)

    X, Y, Z = sgr_matrix.T.dot(np.array([Xs, Ys, Zs]))

    l = np.arctan2(Y, X)*u.radian
    b = np.arcsin(Z/np.sqrt(X*X+Y*Y+Z*Z))*u.radian

    l[l<0] += 2*np.pi*u.radian

    return coord.Galactic(l=l, b=b, distance=orp_coord.distance)
