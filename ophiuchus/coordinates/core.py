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

__all__ = ["Ophiuchus", "R"]

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

@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Ophiuchus)
def galactic_to_oph():
    """ Compute the transformation from Galactic spherical to
        heliocentric Oph coordinates.
    """
    return R

# Oph to Galactic coordinates
@frame_transform_graph.transform(coord.StaticMatrixTransform, Ophiuchus, coord.Galactic)
def oph_to_galactic():
    """ Compute the transformation from heliocentric Oph coordinates to
        spherical Galactic.
    """
    return galactic_to_oph().T
